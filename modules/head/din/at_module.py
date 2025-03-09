import torch.nn as nn
import torch
import math

class Actor_Transformer(nn.Module):
    def __init__(self, in_dim, temporal_pooled_first=False, dropout=0.1):
        super(Actor_Transformer, self).__init__()
        self.in_dim = in_dim
        self.temporal_pooled_first = temporal_pooled_first
        self.Q_W = nn.Linear(in_dim, in_dim, bias=False)
        self.K_W = nn.Linear(in_dim, in_dim, bias=False)
        self.V_W = nn.Linear(in_dim, in_dim, bias=False)
        self.layernorm1 = nn.LayerNorm([in_dim])
        self.dropout1 = nn.Dropout(dropout)

        self.FFN_linear1 = nn.Linear(in_dim, in_dim, bias=True)
        self.FFN_relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_linear2 = nn.Linear(in_dim, in_dim, bias=True)

        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm([in_dim])

    def forward(self, x):
        '''
        :param x: shape [B, T, N, NFB]
        :return:
        '''
        B, T, N, NFB = x.shape
        if self.temporal_pooled_first:
            x = torch.mean(x, dim = 1)
        else:
            x = x.view(B * T, N, NFB)

        query = self.Q_W(x)
        keys = self.K_W(x).transpose(1, 2)
        values = self.V_W(x)
        att_weight = torch.bmm(query, keys) / math.sqrt(self.in_dim)

        att_weight = torch.softmax(att_weight, dim=-1)
        att_values = torch.bmm(att_weight, values)

        x = self.layernorm1(x + self.dropout1(att_values))
        FFN_x = self.FFN_linear1(x)
        FFN_x = self.FFN_relu(FFN_x)
        FFN_x = self.dropout2(FFN_x)
        FFN_x = self.FFN_linear2(FFN_x)
        x = self.layernorm2(x + self.dropout2(FFN_x))
        return x