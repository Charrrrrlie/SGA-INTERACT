import torch
import torch.nn as nn

from modules.head.actionformer.PtTransformer import PtTransformerClsHead, PtTransformerRegHead

class STT_Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(STT_Block, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, src):
        # src: [L, N, E] tgt: [S, N, E]
        output, _ = self.multi_head_attention(tgt, src, src)
        output = self.norm1(output + tgt)
        output = output + self.feed_forward(output)
        output = self.norm2(output)

        return output

class STT_Layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(STT_Layer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.spatio_attention = STT_Block(embed_dim, num_heads)
        self.temporal_attention = STT_Block(embed_dim, num_heads)

        self.spatio_cross_attention = STT_Block(embed_dim, num_heads)
        self.temporal_cross_attention = STT_Block(embed_dim, num_heads)
    
    def forward(self, x):
        B, N, T, C = x.shape

        x_temp = x.permute(2, 0, 1, 3).reshape(T, B * N, C).contiguous()
        x_spatio = x.permute(1, 0, 2, 3).reshape(N, B * T, C).contiguous()

        x_temp = self.spatio_attention(x_temp, x_temp)
        x_spatio = self.temporal_attention(x_spatio, x_spatio)

        x_spatio_feat = x_spatio.reshape(N, B, T, C).permute(2, 1, 0, 3).reshape(T, B * N, C).contiguous()
        x_temp_out = self.temporal_cross_attention(x_temp, x_spatio_feat)


        x_temp_feat = x_temp.reshape(T, B, N, C).permute(2, 1, 0, 3).reshape(N, B * T, C).contiguous()
        x_spatio_out = self.spatio_cross_attention(x_spatio, x_temp_feat)

        x_spatio_out = x_spatio_out.reshape(N, B, T, C).permute(1, 0, 2, 3).contiguous()
        x_temp_out = x_temp_out.reshape(T, B, N, C).permute(1, 2, 0, 3).contiguous()

        return x_temp_out + x_spatio_out

class STT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes, multi_layer_sup=False):
        super(STT, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([STT_Layer(embed_dim, num_heads) for _ in range(num_layers)])
        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.multi_layer_sup = multi_layer_sup

    def forward(self, x):
        # [B, N, T, C]
        x = self.input_projection(x)

        if self.multi_layer_sup:
            outputs = []
            for layer in self.layers:
                x = layer(x)
                outputs.append(self.classifier(x.mean(dim=[1, 2])))
            return outputs

        else:
            for layer in self.layers:
                x = layer(x)

            x = x.mean(dim=[1, 2])

            x = self.classifier(x)

            return x


class STTLoc(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes, multi_layer_sup=False):
        super(STTLoc, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([STT_Layer(embed_dim, num_heads) for _ in range(num_layers)])
        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.cls_head = PtTransformerClsHead(input_dim=embed_dim,
                                             feat_dim=embed_dim,
                                             num_classes=num_classes)
        self.reg_head = PtTransformerRegHead(input_dim=embed_dim,
                                                feat_dim=embed_dim,
                                                output_dim=1)
        self.offset_head = PtTransformerRegHead(input_dim=embed_dim,
                                                feat_dim=embed_dim,
                                                output_dim=1)

        self.multi_layer_sup = multi_layer_sup

    def forward(self, x):
        # [B, N, T, C]
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1).permute(0, 2, 1).contiguous()

        # [B, C, T]
        cls_heatmap = self.cls_head(x) # [B, num_class, T]
        cls_heatmap = torch.sigmoid(cls_heatmap)

        offset_heatmap = self.offset_head(x) # [B, 1, T]
        offset_heatmap = torch.sigmoid(offset_heatmap)

        reg_heatmap = self.reg_head(x) # [B, 1, T]
        reg_heatmap = torch.sigmoid(reg_heatmap)

        return cls_heatmap, offset_heatmap, reg_heatmap
