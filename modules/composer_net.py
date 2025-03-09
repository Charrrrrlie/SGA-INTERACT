import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict

from modules.composer.misc import build_mlp, get_joint_graph
from modules.composer.misc import GraphConvolution
from modules.composer.position_encoding import PositionEmbeddingAbsoluteLearned_1D
from modules.composer.position_encoding import LearnedFourierFeatureTransform
from modules.composer.tnt_four_scales import TNT

from modules.loss_func.loss import compute_multi_weighted_recognition_loss
from modules.loss_func.auxiliary_loss import compute_contrastive_loss
from modules.composer.utils import get_group_feats_source, get_group_feats_fix
from metrics import accuracy

class COMPOSER(nn.Module):
    def __init__(self, model_args, dataset_args):
        super(COMPOSER, self).__init__()
        args = edict()
        args.update(model_args.model)
        args.update({'loss': model_args.loss})
        args.update(dataset_args)
        self.args = args

        self.interaction_indexes = [
            self.args.N*i+j for i in range(self.args.N) 
            for j in range(self.args.N) if self.args.N*i+j != self.args.N*i+i]

        embedding_dim = args.joint_initial_feat_dim
        self.joint_class_embed_layer = nn.Embedding(args.J, embedding_dim)
        gcn_layers = [
            GraphConvolution(in_features=embedding_dim, out_features=embedding_dim,
                             dropout=0, act=F.relu, use_bias=True) 
            for l in range(self.args.num_gcn_layers)] 
        self.joint_class_gcn_layers = nn.Sequential(*gcn_layers)

        self.adj = get_joint_graph(num_nodes=args.J, joint_graph_path=args.joint_graph_path)

        self.special_token_embed_layer = nn.Embedding(args.max_num_tokens, args.TNT_hidden_dim)
        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.max_times_embed, embedding_dim)
        self.image_embed_layer = LearnedFourierFeatureTransform(2, embedding_dim // 2)

        # joint track projection layer
        joint_track_proj_input_dim = args.T * embedding_dim * 4
        if 'Basketball' in self.args.name:
            joint_track_proj_input_dim -= args.T
        self.joint_track_projection_layer = build_mlp(input_dim=joint_track_proj_input_dim, 
                                                hidden_dims=[args.TNT_hidden_dim], 
                                                output_dim=args.TNT_hidden_dim,
                                                use_batchnorm=args.projection_batchnorm,
                                                dropout=args.projection_dropout)

        # person track projection layer
        self.person_track_projection_layer = build_mlp(input_dim=args.J*args.T*self.args.joint_dim,
                                                 hidden_dims=[args.TNT_hidden_dim], 
                                                 output_dim=args.TNT_hidden_dim,
                                                 use_batchnorm=args.projection_batchnorm,
                                                 dropout=args.projection_dropout)

        # person interaction track projection layer
        self.interaction_track_projection_layer = build_mlp(input_dim=args.TNT_hidden_dim*2,
                                                      hidden_dims=[args.TNT_hidden_dim], 
                                                      output_dim=args.TNT_hidden_dim,
                                                      use_batchnorm=args.projection_batchnorm,
                                                      dropout=args.projection_dropout)

        # group track projection layer
        self.person_to_group_projection = build_mlp(input_dim=(args.N//2)*args.TNT_hidden_dim,
                                                    hidden_dims=[args.TNT_hidden_dim], 
                                                    output_dim=args.TNT_hidden_dim,
                                                    use_batchnorm=args.projection_batchnorm,
                                                    dropout=args.projection_dropout)

        # ball track projection layer
        # if hasattr(args, 'ball_trajectory_use') and args.ball_trajectory_use:
        #     self.ball_track_projection_layer = build_mlp(input_dim=(2*args.joint_initial_feat_dim+4)*args.T,
        #                                              hidden_dims=[args.TNT_hidden_dim], 
        #                                              output_dim=args.TNT_hidden_dim,
        #                                              use_batchnorm=args.projection_batchnorm,
        #                                              dropout=args.projection_dropout)

        # TNT blocks
        assert not hasattr(args, 'ball_trajectory_use') or args.ball_trajectory_use == False, \
            'Do not use ball information'

        self.TNT = TNT(args, args.TNT_hidden_dim, args.TNT_n_layers, final_norm=True, return_intermediate=True)
        
        # Prediction
        self.classifier = build_mlp(input_dim=args.TNT_hidden_dim, 
                                    hidden_dims=None, output_dim=args.num_classes, 
                                    use_batchnorm=args.classifier_use_batchnorm, 
                                    dropout=args.classifier_dropout)

        if 'num_person_action_classes' not in args:
            self.person_classifier = None
        else:
            self.person_classifier = build_mlp(input_dim=args.TNT_hidden_dim, 
                                        hidden_dims=None, output_dim=args.num_person_action_classes, 
                                        use_batchnorm=args.classifier_use_batchnorm, 
                                        dropout=args.classifier_dropout)

        # Prototypes
        self.prototypes = nn.Linear(args.TNT_hidden_dim, args.nmb_prototypes, bias=False)

    def get_joint_feats_composite_encoded(self, joint_feats):
        B, N, J, T = joint_feats.shape[:4]
        device = joint_feats.device

        # image coords positional encoding
        # NOTE(yyc): using coord_scale_factor to align with the image size and only use x-y projection plane
        image_coords = (joint_feats[...,-self.args.joint_dim:][..., :2] * self.args.coord_scale_factor).to(torch.int64)
        coords_h = np.linspace(0, 1, self.args.image_h * self.args.coord_scale_factor, endpoint=False)
        coords_w = np.linspace(0, 1, self.args.image_w * self.args.coord_scale_factor, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
        image_coords_learned =  self.image_embed_layer(xy_grid).squeeze(0).permute(1, 2, 0)

        image_coords_embeded = image_coords_learned[image_coords[...,1], image_coords[...,0]]
        # (B, N, J, T, d_0)

        # removing the raw joint coordinates dim
        joint_feats = joint_feats[..., :-self.args.joint_dim]  

        # time positional encoding
        time_ids = torch.arange(1, T+1, device=device).repeat(B, N, J, 1)
        time_seq = self.time_embed_layer(time_ids) 

        # joint classes embedding learning as tokens/nodes
        joint_class_ids = joint_feats[..., -1]  # note that the last dim is the joint class id by default

        joint_classes_embeded = self.joint_class_embed_layer(joint_class_ids.type(torch.cuda.LongTensor)) # (B, N, J, T, d_0)

        joint_classes_embeded = joint_classes_embeded.transpose(2, 3).flatten(0, 1).flatten(0, 1)  # joint_classes_embeded: (B*N*T, J, d_0)
        input = (joint_classes_embeded, self.adj.repeat(B*N*T, 1, 1).to(device))  # adj: # (B*N*T, J, J)
        joint_classes_encode = self.joint_class_gcn_layers(input)[0]
        joint_classes_encode = joint_classes_encode.view(B, N, T, J, -1).transpose(2, 3)  # (B, N, J, T, d_0)

        # removing the joint class dim (last dim by default)
        joint_feats = joint_feats[..., :-1]

        joint_feats_composite_encoded = torch.cat(
            [joint_feats, time_seq, image_coords_embeded, joint_classes_encode], 
            dim=-1) 
        return joint_feats_composite_encoded

    def forward(self, x):

        joint_feats = x['joint_feats']

        B, N, J, T = joint_feats.shape[:4]
        d = self.args.TNT_hidden_dim
        
        joint_feats_for_person = joint_feats[..., :self.args.joint_dim]

        # CLS initial embedding
        CLS_id = torch.arange(1, device=joint_feats.device).repeat(B, 1)
        CLS = self.special_token_embed_layer(CLS_id)

        # PROJECTIONS
        # joint track projection
        joint_feats_composite_encoded = self.get_joint_feats_composite_encoded(joint_feats)
        joint_track_feats_proj = self.joint_track_projection_layer(
            joint_feats_composite_encoded.flatten(3, 4).flatten(0, 1).flatten(0, 1)  # (B*N*J, T*d_0)
        ).view(B, N*J, -1)
        # (B, N*J, d)

        # person track projection
        person_track_feats_proj = self.person_track_projection_layer(
            joint_feats_for_person.flatten(0, 1).contiguous().view(B*N, -1)
        ).view(B, N, -1)
        # (B, N, d)
        
        # form sequence of person-person-interaction-track tokens
        tem1 = person_track_feats_proj.repeat(1, N, 1).reshape(B,N,N,d).transpose(1, 2).flatten(1, 2)  # (B, N^2, d)
        tem2 = person_track_feats_proj.repeat(1, N, 1) # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interaction_track_feats = tem3[:, self.interaction_indexes, :]  # (B, N*(N-1), 2*d)
        interaction_track_feats_proj = self.interaction_track_projection_layer(
            interaction_track_feats.flatten(0, 1)).view(B, N*(N-1), -1)  # (B, N*(N-1), d)

        # obtain person to group mapping
        if 'Volleyball' in self.args.name:
            people_middle_hip_coords = (
                joint_feats_for_person[:,:,11,self.args.group_person_frame_idx,-2:] + 
                joint_feats_for_person[:,:,12,self.args.group_person_frame_idx,-2:]) / 2
            # (B, N, 2)  - W, H (X, Y)

            people_idx_sort_by_middle_hip_xcoord = torch.argsort(people_middle_hip_coords[:,:,0], dim=-1)  # (B, N)
            left_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, :int(self.args.N//2)]  # (B, N/2)
            right_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, int(self.args.N//2):]  # (B, N/2)

        elif 'Basketball' in self.args.name:
            left_group_people_idx = x['host_team_info']
            right_group_people_idx = x['guest_team_info']
        else:
            raise ValueError('Please check the dataset name!')

        if self.args.source_version:
            left_group_people_repre, right_group_people_repre = get_group_feats_source(person_track_feats_proj,
                                                                                    left_group_people_idx,
                                                                                    right_group_people_idx,
                                                                                    self.args.N//2)
        else:
            left_group_people_repre, right_group_people_repre = get_group_feats_fix(person_track_feats_proj,
                                                                                    left_group_people_idx,
                                                                                    right_group_people_idx)

        left_group_feats_proj = self.person_to_group_projection(left_group_people_repre.flatten(1,2))   # (B, d)
        right_group_feats_proj = self.person_to_group_projection(right_group_people_repre.flatten(1,2))   # (B, d)
        group_track_feats_proj = torch.stack([left_group_feats_proj, right_group_feats_proj], dim=1)  # (B, 2, d)

        outputs = self.TNT(CLS.transpose(0, 1),  # (1, B, d)
                            joint_track_feats_proj.transpose(0, 1),  # (N*J, B, d)
                            person_track_feats_proj.transpose(0, 1),  # (N, B, d)
                            interaction_track_feats_proj.transpose(0, 1),  # (N*(N-1), B, d)
                            group_track_feats_proj.transpose(0, 1),  # (2, B, d)
                            left_group_people_idx,
                            right_group_people_idx
                            )
               
        # outputs is a list of list
        # len(outputs) is the numbr of TNT layers
        # each inner list is [CLS_f, CLS_m, CLS_c, output_CLS, output_fine, output_middle, output_coarse, output_group]

        # CLASSIFIER
        pred_logits = []
        for l in range(self.args.TNT_n_layers):
            
            fine_cls = outputs[l][0].transpose(0, 1).squeeze(1)  # (B, d)
            middle_cls = outputs[l][1].transpose(0, 1).squeeze(1)  # (B, d)
            coarse_cls = outputs[l][2].transpose(0, 1).squeeze(1)  # (B, d)
            group_cls = outputs[l][3].transpose(0, 1).squeeze(1)  # (B, d)
            
            pred_logit_f = self.classifier(fine_cls)
            pred_logit_m = self.classifier(middle_cls)
            pred_logit_c = self.classifier(coarse_cls)
            pred_logit_g = self.classifier(group_cls)
            
            pred_logits.append([pred_logit_f, pred_logit_m, pred_logit_c, pred_logit_g])
            
             
        # fine_cls, middle_cls, coarse_cls, group_cls are from the last layer
        fine_cls_normed = nn.functional.normalize(fine_cls, dim=1, p=2)
        middle_cls_normed = nn.functional.normalize(middle_cls, dim=1, p=2)
        coarse_cls_normed = nn.functional.normalize(coarse_cls, dim=1, p=2)
        group_cls_normed = nn.functional.normalize(group_cls, dim=1, p=2)

        scores_f = self.prototypes(fine_cls_normed)
        scores_m = self.prototypes(middle_cls_normed)
        scores_c = self.prototypes(coarse_cls_normed)
        scores_g = self.prototypes(group_cls_normed)
        scores = [scores_f, scores_m, scores_c, scores_g]

        # PERSON CLASSIFIER
        pred_logits_person = []
        if self.person_classifier is not None:
            for l in range(self.args.TNT_n_layers):
                person_feats = outputs[l][5].transpose(0, 1).flatten(0,1)  # (BxN, d)
                pred_logit_person = self.person_classifier(person_feats)  
                pred_logits_person.append(pred_logit_person)

        # -------------------------- Loss --------------------------
        loss_dict = {}
        contrastive_loss = compute_contrastive_loss(scores,
                                                    self.args.loss.cluster_assignment.loss_coe_constrastive_clustering,
                                                    self.args.loss.cluster_assignment.sinkhorn_iterations,
                                                    self.args.loss.cluster_assignment.temperature)
        loss_dict['contrastive_clustering_loss'] = contrastive_loss
        
        group_label = x['group_label']
        person_labels = x['person_labels'].flatten() if 'person_labels' in x else None
        group_loss, person_loss = compute_multi_weighted_recognition_loss(self.args.loss,
                                                                          pred_logits,
                                                                          group_label,
                                                                          pred_logits_person if 'person_labels' in x else None,
                                                                          person_labels)
        loss_dict['group_loss'] = group_loss
        if self.person_classifier is not None:
            loss_dict['person_loss'] = person_loss

        prec1, prec3 = accuracy(pred_logits[-1][-1], group_label, topk=(1, 3))
        info_dict = {
            'group_acc1': prec1,
            'group_acc3': prec3,
        }
        if 'person_labels' in x:
            prec1_person, prec3_person = accuracy(pred_logits_person[-1], person_labels, topk=(1, 3))
            prec1_person = prec1_person.reshape(B, -1).mean(dim=1)
            prec3_person = prec3_person.reshape(B, -1).mean(dim=1)
            info_dict.update({
                'person_acc1': prec1_person if 'person_labels' in x else 0.0,
                'person_acc3': prec3_person if 'person_labels' in x else 0.0,
            })

        if 'record_per_item_stats' in x and x['record_per_item_stats']:
            res_info_dict = {}
            res_info_dict['pred_group'] = torch.argmax(pred_logits[-1][-1], dim=1)
            if 'person_labels' in x:
                res_info_dict['pred_person'] = torch.argmax(pred_logits_person[-1], dim=1)
            return loss_dict, info_dict, res_info_dict

        return loss_dict, info_dict