import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.head.din.at_module import Actor_Transformer
from modules.head.actionformer.PtTransformer import PtTransformerClsHead, PtTransformerRegHead

class AT(nn.Module):
    def __init__(self,
                 num_input_features,
                 num_features_boxes,
                 num_classes):
        super(AT, self).__init__()

        NFI = num_input_features
        NFB = num_features_boxes

        self.fc_emb_1 = nn.Linear(NFI, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # AT inference
        self.AT = Actor_Transformer(NFB)
        self.fc_activities = nn.Linear(NFB, num_classes)

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
        boxes_features = F.relu(boxes_features, inplace=True)

        # AT inference
        boxes_states = self.AT(boxes_features)
        torch.cuda.empty_cache()

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return activities_scores

class ATLoc(AT):
    def __init__(self,
                 num_input_features,
                 num_features_boxes,
                 num_classes):
        super(ATLoc, self).__init__(num_input_features,
                                    num_features_boxes,
                                    num_classes)
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
        boxes_features = F.relu(boxes_features, inplace=True)

        # AT inference
        boxes_states = self.AT(boxes_features)
        torch.cuda.empty_cache()

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
        boxes_states_pooled = boxes_states_pooled.view(B, T, -1).permute(0, 2, 1).contiguous()

        # [B, N, C]
        cls_heatmap = self.cls_head(boxes_states_pooled) # [B, num_class, T]
        cls_heatmap = torch.sigmoid(cls_heatmap)

        offset_heatmap = self.offset_head(boxes_states_pooled) # [B, 1, T]
        offset_heatmap = torch.sigmoid(offset_heatmap)

        reg_heatmap = self.reg_head(boxes_states_pooled) # [B, 1, T]
        reg_heatmap = torch.sigmoid(reg_heatmap)

        return cls_heatmap, offset_heatmap, reg_heatmap