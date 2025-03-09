import torch
import torch.nn as nn

from easydict import EasyDict as edict

import modules.head

from modules.backbone.graph.openpose_graph import Graph
from modules.backbone.st_gcn import STGCN
from modules.loss_func.loss import compute_recognition_loss
from modules.loss_func.loc_loss import compute_focal_loss, compute_diou_loss_1d

from modules.composer.position_encoding import PositionEmbeddingAbsoluteLearned_1D

from metrics import accuracy

def load_stgcn_backbone_pretrained(backbone, ckpt_path, config):
    ckpt = torch.load(ckpt_path)
    for k in list(ckpt.keys()):
        if 'person_bn' in k or 'fcn' in k:
            ckpt.pop(k)
            continue
        if 'module.backbone.' in k and config is not None:
            if int(k.split('.')[2]) >= len(config):
                ckpt.pop(k)
                continue
        ckpt[k.replace('module.', '')] = ckpt.pop(k)

    for k in list(ckpt.keys()):
        if 'data_bn' in k:
            # NOTE(yyc): we only use the first person
            ckpt[k] = ckpt[k][:backbone.kp_dim * backbone.num_point]
    backbone.load_state_dict(ckpt)
    return backbone

class ONE2MANY(nn.Module):
    def __init__(self, model_args, dataset_args):
        super(ONE2MANY, self).__init__()
        args = edict()

        args.update(model_args.model)
        args.update({'loss': model_args.loss})
        args.update(dataset_args)
        self.args = args

        assert args.backbone.name == 'stgcn', 'Only support ST-GCN backbone now'

        graph = Graph(**args.backbone.graph_args)
        self.backbone = STGCN(graph=graph, **args.backbone.params)

        if args.backbone.pretrained:
            backbone_config = args.backbone.params.backbone_config if 'backbone_config' in args.backbone.params else None
            self.backbone = load_stgcn_backbone_pretrained(self.backbone,
                                                           args.backbone.ckpt_path,
                                                           backbone_config)
            # init self.backbone.data_bn
            if 'no_reset_bn' not in args.backbone or not args.backbone.no_reset_bn:
                self.backbone.data_bn.weight.data.fill_(1)
                self.backbone.data_bn.bias.data.zero_()

        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.time_embed.max_times_embed, args.time_embed.embed_dim)

        if 'team_info_embed' in args:
            self.team_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.team_info_embed.max_team_embed, args.team_info_embed.embed_dim)
        else:
            self.team_embed_layer = None

        if args.ball_trajectory_use:
            assert 'ball_embed' in args, 'Ball trajectory is used, but ball_embed must be provided'
            self.ball_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.ball_embed.max_ball_embed, args.ball_embed.embed_dim)
        else:
            self.ball_embed_layer = None

        # extra processing for DIN and ARG head
        if args.head.name == 'DIN' or args.head.name == 'ARG':
            args.head.params['num_frames'] = self.args.backbone.params.window_size // self.args.backbone.downsampling
            args.head.params['num_boxes'] = self.args.N

        self.head = getattr(modules.head, args.head.name)(**args.head.params, num_classes=args.num_classes)

    def forward(self, x):

        joint_feats = x['joint_feats']

        B, N, C, T, J = joint_feats.shape
        device = joint_feats.device

        # time encoding in [B, Ct, T // downsampling]
        time_ids = torch.arange(1, T // self.args.backbone.downsampling + 1, device=device).repeat(B, 1)
        time_embed = self.time_embed_layer(time_ids).permute(0, 2, 1)

        # ball embedding in [B, N, Ct, T // downsampling]
        if self.ball_embed_layer is not None:
            ball_info = x['ball_occupy_feats'][:, ::self.args.backbone.downsampling, :].to(torch.long)
            ball_embed = self.ball_embed_layer(ball_info).permute(0, 2, 3, 1)

        # extract features per-person in [B, N, window_size // downsampling, Cb]
        backbone_feats = []
        for player in range(N):
            feature_channel = (T - self.args.backbone.params.window_size) // self.args.backbone.stride
            backbone_person_feats = []
            for i_feat in range(feature_channel + 1):
                start_frame = i_feat * self.args.backbone.stride
                end_frame = start_frame + self.args.backbone.params.window_size
                backbone_person_feat = self.backbone(joint_feats[:, player, :, start_frame:end_frame, :])

                start_frame = start_frame // self.args.backbone.downsampling
                end_frame = end_frame // self.args.backbone.downsampling

                if self.ball_embed_layer is not None:
                    backbone_person_feat = backbone_person_feat + ball_embed[:, player, :, start_frame:end_frame]

                if 'fusion_type' in self.args.time_embed and self.args.time_embed.fusion_type == 'add':
                    backbone_person_feats.append(backbone_person_feat + time_embed[..., start_frame:end_frame])
                else:
                    backbone_person_feats.append(torch.cat([backbone_person_feat,
                                                            time_embed[..., start_frame:end_frame]],
                                                            dim=1))

            backbone_person_feats = torch.cat(backbone_person_feats, dim=1)
            backbone_feats.append(backbone_person_feats)
        backbone_feats = torch.stack(backbone_feats, dim=1)
        backbone_feats = backbone_feats.permute(0, 1, 3, 2).contiguous()

        # team embedding
        if self.team_embed_layer is not None:
            # set host team players to 1, guest team players to 0
            team_info = torch.zeros([B, N], dtype=torch.long).to(device)
            host_team = x['host_team_info']
            guest_team = x['guest_team_info']
            team_info = torch.scatter(team_info, 1, host_team, 1)

            team_embed = self.team_embed_layer(team_info).view(B, N, 1, -1)
            # backbone_feats = backbone_feats + team_embed
            team_embed = team_embed.repeat(1, 1, backbone_feats.shape[2], 1)
            backbone_feats = torch.cat([backbone_feats, team_embed], dim=-1)

        # head
        pred_logits = self.head(backbone_feats)

        # -------------------------- Loss --------------------------
        loss_dict = {}

        group_label = x['group_label']
        loss_dict['group_loss'] = compute_recognition_loss(self.args.loss, pred_logits, group_label)

        if isinstance(pred_logits, list):
            pred_logits = pred_logits[-1]

        prec1, prec3 = accuracy(pred_logits, group_label, topk=(1, 3))
        info_dict = {
            'group_acc1': prec1,
            'group_acc3': prec3,
        }

        if 'record_per_item_stats' in x and x['record_per_item_stats']:
            res_info_dict = {}
            res_info_dict['pred_group'] = torch.argmax(pred_logits, dim=1)
            return loss_dict, info_dict, res_info_dict

        return loss_dict, info_dict


def decode_heatmap(cls_heatmap, offset_heatmap, reg_heatmap, K=5):
    B, C, T_sampled = cls_heatmap.shape
    device = cls_heatmap.device

    topk_scores, topk_inds = torch.topk(cls_heatmap, K, dim=-1)

    topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K, dim=-1)

    pred_ind = torch.gather(topk_inds.view(B, -1), 1, topk_ind)
    pred_cls = topk_ind // K
    pred_score = topk_score

    offset = torch.gather(offset_heatmap.squeeze(1), -1, pred_ind)
    reg = torch.gather(reg_heatmap.squeeze(1), -1, pred_ind) * (T_sampled // 2)

    segment = torch.stack([pred_ind + offset - reg, pred_ind + offset + reg], dim=-1)
    segment = torch.clamp(segment, 1e-4, T_sampled - 1e-4)

    return pred_cls, pred_score, segment

def decode_heatmap_from_gt(gt_segment, offset_heatmap, reg_heatmap, T_sampled):
    gt_centers = (gt_segment[:, :, 0] + gt_segment[:, :, 1]) / 2
    gt_inds = gt_centers.long()
    pred_segment = torch.zeros_like(gt_segment).to(gt_segment.device)

    pred_offset = torch.gather(offset_heatmap.squeeze(1), -1, gt_inds)
    pred_reg = torch.gather(reg_heatmap.squeeze(1), -1, gt_inds) * (T_sampled // 2)

    pred_segment[..., 0] = gt_inds + pred_offset - pred_reg
    pred_segment[..., 1] = gt_inds + pred_offset + pred_reg

    pred_segment = torch.clamp(pred_segment, 1e-4, T_sampled - 1e-4)

    pred_segment[gt_centers == 0] = 0

    return pred_segment

class ONE2MANYLOC(nn.Module):
    def __init__(self, model_args, dataset_args):
        super(ONE2MANYLOC, self).__init__()
        args = edict()

        args.update(model_args.model)
        args.update({'loss': model_args.loss})
        args.update(dataset_args)
        self.args = args

        assert args.backbone.name == 'stgcn', 'Only support ST-GCN backbone now'

        graph = Graph(**args.backbone.graph_args)
        self.backbone = STGCN(graph=graph, **args.backbone.params)

        if args.backbone.pretrained:
            backbone_config = args.backbone.params.backbone_config if 'backbone_config' in args.backbone.params else None
            self.backbone = load_stgcn_backbone_pretrained(self.backbone,
                                                           args.backbone.ckpt_path,
                                                           backbone_config)
            # init self.backbone.data_bn
            if 'no_reset_bn' not in args.backbone or not args.backbone.no_reset_bn:
                self.backbone.data_bn.weight.data.fill_(1)
                self.backbone.data_bn.bias.data.zero_()

        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.time_embed.max_times_embed, args.time_embed.embed_dim)

        if 'team_info_embed' in args:
            self.team_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.team_info_embed.max_team_embed, args.team_info_embed.embed_dim)
        else:
            self.team_embed_layer = None

        if args.ball_trajectory_use:
            assert 'ball_embed' in args, 'Ball trajectory is used, but ball_embed must be provided'
            self.ball_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.ball_embed.max_ball_embed, args.ball_embed.embed_dim)
        else:
            self.ball_embed_layer = None

        # extra processing for DIN and ARG head
        if args.head.name == 'DINLoc' or args.head.name == 'ARGLoc':
            args.head.params['num_frames'] = self.args.T // self.args.backbone.downsampling
            args.head.params['num_boxes'] = self.args.N
        self.head = getattr(modules.head, args.head.name)(**args.head.params, num_classes=args.num_classes)

    def forward(self, x):

        joint_feats = x['joint_feats']

        B, N, C, T, J = joint_feats.shape
        device = joint_feats.device

        # time encoding in [B, Ct, T // downsampling]
        time_ids = torch.arange(1, T // self.args.backbone.downsampling + 1, device=device).repeat(B, 1)
        time_embed = self.time_embed_layer(time_ids).permute(0, 2, 1)
        assert self.args.time_embed.fusion_type == 'add', 'Only support add fusion now'

        # ball embedding in [B, N, Ct, T // downsampling]
        if self.ball_embed_layer is not None:
            ball_info = x['ball_occupy_feats'][:, ::self.args.backbone.downsampling, :].to(torch.long)
            ball_embed = self.ball_embed_layer(ball_info).permute(0, 2, 3, 1)

        # extract features per-person in [B, N, window_size // downsampling, Cb]
        backbone_feats = torch.zeros(B, N, self.args.backbone.feature_dim, T // self.args.backbone.downsampling).to(device)
        backbone_feat_overlap = torch.zeros(1, 1, 1, T // self.args.backbone.downsampling).to(device)
        for player in range(N):
            feature_channel = (T - self.args.backbone.params.window_size) // self.args.backbone.stride
            for i_feat in range(feature_channel + 1):
                start_frame = i_feat * self.args.backbone.stride
                end_frame = start_frame + self.args.backbone.params.window_size
                backbone_person_feat = self.backbone(joint_feats[:, player, :, start_frame:end_frame, :])

                start_frame = start_frame // self.args.backbone.downsampling
                end_frame = end_frame // self.args.backbone.downsampling

                if self.ball_embed_layer is not None:
                    backbone_person_feat = backbone_person_feat + ball_embed[:, player, :, start_frame:end_frame]

                backbone_feats[:, player, :, start_frame:end_frame] += backbone_person_feat
                if player == 0:
                    backbone_feat_overlap[:, :, :, start_frame:end_frame] += 1

        backbone_feats = backbone_feats / backbone_feat_overlap
        backbone_feats = backbone_feats + time_embed.unsqueeze(1)
        backbone_feats = backbone_feats.permute(0, 1, 3, 2).contiguous()

        # team embedding
        if self.team_embed_layer is not None:
            # set host team players to 1, guest team players to 0
            team_info = torch.zeros([B, N], dtype=torch.long).to(device)
            host_team = x['host_team_info']
            guest_team = x['guest_team_info']
            team_info = torch.scatter(team_info, 1, host_team, 1)

            team_embed = self.team_embed_layer(team_info).view(B, N, 1, -1)
            if self.args.team_info_embed.fusion_type == 'add':
                backbone_feats = backbone_feats + team_embed
            else:
                team_embed = team_embed.repeat(1, 1, backbone_feats.shape[2], 1)
                backbone_feats = torch.cat([backbone_feats, team_embed], dim=-1)

        # head
        pred_cls_heatmap, pred_offset_heatmap, pred_reg_heatmap = self.head(backbone_feats)
        pred_cls, pred_scores, pred_segment = decode_heatmap(pred_cls_heatmap, pred_offset_heatmap, pred_reg_heatmap)

        # -------------------------- Loss --------------------------
        gt_cls_heatmap, gt_valid_mask = self.assign_targets(x,
                                                            T // self.args.backbone.downsampling,
                                                            self.args.backbone.downsampling)

        loss_dict = {}

        loss_dict['group_cls_loss'] = compute_focal_loss(pred_cls_heatmap, gt_cls_heatmap) * self.args.loss.loss_cls

        # regression loss
        gt_segment = x['group_segment']
        decode_segment = decode_heatmap_from_gt(gt_segment / self.args.backbone.downsampling,
                                                pred_offset_heatmap,
                                                pred_reg_heatmap,
                                                T_sampled= T // self.args.backbone.downsampling)
        decode_segment = decode_segment * self.args.backbone.downsampling

        loss_dict['group_reg_loss'] = compute_diou_loss_1d(decode_segment, gt_segment, gt_valid_mask) * self.args.loss.loss_reg

        info_dict = {
            'group_cls': pred_cls,
            'group_cls_confidence': pred_scores,
            'group_segment': pred_segment * self.args.backbone.downsampling,
        }

        return loss_dict, info_dict

    def assign_targets(self, x, T, downsample, radius=15):
        gt_class = x['group_label']
        gt_segment = x['group_segment']
        gt_center = ((gt_segment[:, :, 0] + gt_segment[:, :, 1]) / 2).int().float()

        B, N = gt_class.shape
        device = gt_class.device

        gt_cls_heatmap = torch.zeros(B, self.args.num_classes, T).to(device)

        def draw_gaussian(T, center, sigma, device):
            # returns 1D gaussian heatmaps
            kernel = torch.arange(T, device=device)
            heatmap = torch.exp(-((kernel - center) ** 2) / 2 / sigma ** 2)
            return heatmap

        for i in range(B):
            for j in range(N):
                if gt_class[i, j] == -1:
                    continue
                cls = gt_class[i, j]
                center = gt_center[i, j] // downsample
                gt_cls_heatmap[i, cls] = torch.max(gt_cls_heatmap[i, cls], draw_gaussian(T, center, radius, device))

        gt_valid_mask = gt_class == -1

        return gt_cls_heatmap, gt_valid_mask