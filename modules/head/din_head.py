import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.head.din.dynamic_module import Multi_Dynamic_Inference
from modules.head.actionformer.PtTransformer import PtTransformerClsHead, PtTransformerRegHead

class DIN(nn.Module):
    def __init__(self,
                 num_frames,
                 num_boxes,
                 num_input_features,
                 num_features_boxes,
                 num_classes,
                 dynamic_configs,
                 lite_dim=None,
                 dropout_prob=0.3):
        super(DIN, self).__init__()

        T, N = num_frames, num_boxes
        NFI = num_input_features
        NFB = num_features_boxes

        self.fc_emb_1 = nn.Linear(NFI, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        if lite_dim:
            in_dim = lite_dim
        else:
            in_dim = NFB

        self.lite_dim = lite_dim

        dynamic_configs['in_dim'] = in_dim
        dynamic_configs['T'] = T
        dynamic_configs['N'] = N

        self.DPI = Multi_Dynamic_Inference(**dynamic_configs)

        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=dropout_prob)

        # Lite Dynamic inference
        if self.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, num_classes)
        else:
            self.fc_activities=nn.Linear(NFB, num_classes)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, boxes_features):
        # [B, N, T, C]
        boxes_features = boxes_features.permute(0, 2, 1, 3)

        B, T, N, C = boxes_features.size()
        boxes_features = self.fc_emb_1(boxes_features)
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        if self.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace=True)

        # Dynamic graph inference
        graph_boxes_features = self.DPI(boxes_features)
        torch.cuda.empty_cache()

        # NOTE(yyc): use DIN's vgg16 setting
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        boxes_features = boxes_features.reshape(B, T, N, -1)
        boxes_states = graph_boxes_features + boxes_features
        boxes_states = self.dpi_nl(boxes_states)
        boxes_states = F.relu(boxes_states, inplace=True)
        boxes_states = self.dropout_global(boxes_states)

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)

        # Temporal fusion
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B, -1)

        return activities_scores


class DINLoc(DIN):
    def __init__(self,
                 num_frames,
                 num_boxes,
                 num_input_features,
                 num_features_boxes,
                 num_classes,
                 dynamic_configs,
                 lite_dim=None,
                 dropout_prob=0.3):
        super(DINLoc, self).__init__(num_frames, num_boxes, num_input_features, num_features_boxes, num_classes, dynamic_configs, lite_dim, dropout_prob)
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
        boxes_features = self.fc_emb_1(boxes_features)
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        if self.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace=True)

        # Dynamic graph inference
        graph_boxes_features = self.DPI(boxes_features)
        torch.cuda.empty_cache()

        # NOTE(yyc): use DIN's vgg16 setting
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        boxes_features = boxes_features.reshape(B, T, N, -1)
        boxes_states = graph_boxes_features + boxes_features
        boxes_states = self.dpi_nl(boxes_states)
        boxes_states = F.relu(boxes_states, inplace=True)
        boxes_states = self.dropout_global(boxes_states)

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled = boxes_states_pooled.permute(0, 2, 1).contiguous()

        # [B, C, T]
        cls_heatmap = self.cls_head(boxes_states_pooled) # [B, num_class, T]
        cls_heatmap = torch.sigmoid(cls_heatmap)

        offset_heatmap = self.offset_head(boxes_states_pooled) # [B, 1, T]
        offset_heatmap = torch.sigmoid(offset_heatmap)

        reg_heatmap = self.reg_head(boxes_states_pooled) # [B, 1, T]
        reg_heatmap = torch.sigmoid(reg_heatmap)

        return cls_heatmap, offset_heatmap, reg_heatmap