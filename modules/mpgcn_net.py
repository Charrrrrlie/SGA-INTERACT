import torch
import torch.nn as nn

from easydict import EasyDict as edict

from modules.mpgcn.graphs import Graph
from modules.mpgcn.blocks import Basic_Block, Input_Branch
from modules.loss_func.loss import compute_recognition_loss

from metrics import accuracy

def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, Basic_Block):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)


class MPGCN(nn.Module):
    def __init__(self, model_args, dataset_args):
        super(MPGCN, self).__init__()

        args = edict()
        args.update(model_args)
        args.update({'loss': model_args.loss})
        args.update(dataset_args)
        self.args = args

        graph = Graph(**args.graph)
        A = torch.tensor(graph.A, dtype=torch.float32)
        kwargs = {
            'parts': graph.parts,
        }

        kwargs.update(args.model)
        self.datashape = self.get_datashape(graph)
        num_input, num_channel, _, _, _ = self.datashape

        # input branches
        self.input_branches = nn.ModuleList([
            Input_Branch(num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [
            Basic_Block(32*num_input, 128, A, stride=2, **kwargs),
            Basic_Block(128, 128, A, **kwargs),
            Basic_Block(128, 128, A, **kwargs),
            Basic_Block(128, 256, A, stride=2, **kwargs),
            Basic_Block(256, 256, A, **kwargs),
            Basic_Block(256, 256, A, **kwargs)
        ]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, args.num_classes)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def get_datashape(self, graph):
        I = len(self.args.inputs) if self.args.inputs.isupper() else 1
        C = self.args.input_dims * 2
        T = len(range(*self.args.window)) if 'window' in self.args else self.args.T
        V = graph.num_node
        M = self.args.N // graph.num_person
        return [I, C, T, V, M]

    def forward(self, x):

        joint_feats = x['joint_feats']
        assert list(joint_feats.shape[1:]) == self.datashape, 'data_new.shape: {} {}'.format(joint_feats.shape, self.datashape)

        B, I, C, T, V, M = joint_feats.size()

        # input branches
        joint_feat_cat = []
        for i, branch in enumerate(self.input_branches):
            joint_feat_cat.append(branch(joint_feats[:, i, ...]))
        joint_feats = torch.cat(joint_feat_cat, dim=1)

        # main stream
        for layer in self.main_stream:
            joint_feats = layer(joint_feats)

        # output
        joint_feats = self.global_pooling(joint_feats)
        joint_feats = joint_feats.view(B, M, -1).mean(dim=1)
        pred_logits = self.fcn(joint_feats) # [N, num_classes]

        # -------------------------- Loss --------------------------
        loss_dict = {}

        group_label = x['group_label']
        loss_dict['group_loss'] = compute_recognition_loss(self.args.loss, pred_logits, group_label)

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