# fix bugs in the COMPOSER module
# Detailed infor: https://github.com/hongluzhou/composer/issues/14

import torch
# person_feats_proj [B, N, C]
# _group_people_idx [B, N/2]

def get_group_feats_source(person_feats_proj, left_group_people_idx, right_group_people_idx, num_person_per_group):
    B = person_feats_proj.size(0)
    # form sequence of group track tokens
    left_group_people_repre = person_feats_proj.flatten(
        0,1)[left_group_people_idx.flatten(0,1)].view(B, num_person_per_group, -1)  # (B, N/2, d)
    right_group_people_repre = person_feats_proj.flatten(
        0,1)[right_group_people_idx.flatten(0,1)].view(B, num_person_per_group, -1)  # (B, N/2, d)

    return left_group_people_repre, right_group_people_repre

def get_group_feats_fix(person_feats_proj, left_group_people_idx, right_group_people_idx):
    _, _, C = person_feats_proj.shape

    if left_group_people_idx.dim() == 2:
        left_group_people_idx = left_group_people_idx.unsqueeze(2).expand(-1, -1, C)
        right_group_people_idx = right_group_people_idx.unsqueeze(2).expand(-1, -1, C)

    left_group_people_repre = torch.gather(person_feats_proj, 1, left_group_people_idx)
    right_group_people_repre = torch.gather(person_feats_proj, 1, right_group_people_idx)

    return left_group_people_repre, right_group_people_repre