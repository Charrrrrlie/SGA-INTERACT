
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B, N, D]
        Y: [B, M, D]
    Returns:
        dist: [B, N, M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1,2)) + ry.transpose(1, 2)
    
    return torch.sqrt(dist)

class ARGCN(nn.Module):
    def __init__(self,
                 num_feature_relation,
                 num_features_gcn,
                 num_graph_layer,
                 num_boxes,
                 num_frames,
                 OH=23,
                 OW=40,
                 pos_threshold=0.2):
        super(ARGCN, self).__init__()

        NFR = num_feature_relation
        NG = num_graph_layer
        N = num_boxes
        T = num_frames

        NFG = num_features_gcn

        self.NFR = NFR
        self.NG = NG
        self.OH = OH
        self.OW = OW
        self.pos_threshold = pos_threshold

        self.fc_rn_theta_list = nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = nn.ModuleList([nn.Linear(NFG, NFG, bias=False) for i in range(NG)])

        self.nl_gcn_list = nn.ModuleList([nn.LayerNorm([T * N, NFG]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat=None):
        """
        graph_boxes_features  [B * T, N, NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape

        # Prepare position mask
        if boxes_in_flat is not None:
            graph_boxes_positions = boxes_in_flat  # B * T * N, 4
            graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
            graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
            graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B, N, 2)  # B * T, N, 2

            graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B, N, N

            position_mask = (graph_boxes_distances > (self.pos_threshold * self.OW))
        else:
            position_mask = None

        relation_graph = None
        graph_boxes_features_list = []
        for i in range(self.NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B, N, NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B, N, NFR

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph / math.sqrt(self.NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B * N * N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)

            if position_mask is not None:
                relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features, inplace=True)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list), dim=0)  # B, N, NFG

        return graph_boxes_features, relation_graph