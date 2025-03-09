import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.head.din.arg_module import ARGCN
from modules.head.actionformer.PtTransformer import PtTransformerClsHead, PtTransformerRegHead

class ARG(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self,
                 num_frames,
                 num_boxes,
                 num_input_features,
                 num_features_boxes,
                 num_features_gcn,
                 num_feature_relation,
                 num_graph_layer,
                 num_gcn_layer,
                 num_classes,
                 dropout_prob=0.3):
        super(ARG, self).__init__()

        NFI = num_input_features
        NFB = num_features_boxes

        self.fc_emb_1 = nn.Linear(NFI, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([ARGCN(num_feature_relation,
                                                   num_features_gcn,
                                                   num_graph_layer,
                                                   num_boxes,
                                                   num_frames) for i in range(num_gcn_layer)])
        self.dropout_global = nn.Dropout(p=dropout_prob)
        self.fc_activities = nn.Linear(num_features_gcn, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, boxes_features):
        # [B, N, T, C]
        boxes_features = boxes_features.permute(0, 2, 1, 3)

        B, T, N, C = boxes_features.size()

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features.reshape(B, T * N, -1)
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, None)

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        boxes_features = boxes_features.reshape(B, T, N, -1)

        boxes_states = graph_boxes_features + boxes_features
        boxes_states = self.dropout_global(boxes_states)

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        activities_scores = self.fc_activities(boxes_states_pooled)

        # Temporal fusion
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return activities_scores


class ARGLoc(ARG):
    def __init__(self,
                 num_frames,
                 num_boxes,
                 num_input_features,
                 num_features_boxes,
                 num_features_gcn,
                 num_feature_relation,
                 num_graph_layer,
                 num_gcn_layer,
                 num_classes,
                 dropout_prob=0.3):
        super(ARGLoc, self).__init__(num_frames,
                                    num_boxes,
                                    num_input_features,
                                    num_features_boxes,
                                    num_features_gcn,
                                    num_feature_relation,
                                    num_graph_layer,
                                    num_gcn_layer,
                                    num_classes,
                                    dropout_prob)
        
        self.cls_head = PtTransformerClsHead(input_dim=num_features_boxes,
                                             feat_dim=num_features_boxes,
                                             num_classes=num_classes)
        self.reg_head = PtTransformerRegHead(input_dim=num_features_boxes,
                                                feat_dim=num_features_boxes,
                                                output_dim=1)
        self.offset_head = PtTransformerRegHead(input_dim=num_features_boxes,
                                                feat_dim=num_features_boxes,
                                                output_dim=1)
        del self.fc_activities


    def forward(self, boxes_features):
        # [B, N, T, C]
        boxes_features = boxes_features.permute(0, 2, 1, 3)

        B, T, N, C = boxes_features.size()

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features.reshape(B, T * N, -1)
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, None)

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        boxes_features = boxes_features.reshape(B, T, N, -1)

        boxes_states = graph_boxes_features + boxes_features
        boxes_states = self.dropout_global(boxes_states)

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled = boxes_states_pooled.permute(0, 2, 1).contiguous()

        # [B, N, C]
        cls_heatmap = self.cls_head(boxes_states_pooled) # [B, num_class, T]
        cls_heatmap = torch.sigmoid(cls_heatmap)

        offset_heatmap = self.offset_head(boxes_states_pooled) # [B, 1, T]
        offset_heatmap = torch.sigmoid(offset_heatmap)

        reg_heatmap = self.reg_head(boxes_states_pooled) # [B, 1, T]
        reg_heatmap = torch.sigmoid(reg_heatmap)

        return cls_heatmap, offset_heatmap, reg_heatmap