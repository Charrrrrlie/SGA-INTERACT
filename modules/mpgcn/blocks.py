import torch
from torch import nn

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, use_att=True, stride=1, kernel_size=[9,2], **kwargs):
        super(Basic_Block, self).__init__()
        
        if not len(kernel_size) == 2:
            raise ValueError('Error: Please check whether len(kernel_size) == 2')
        if not kernel_size[0] % 2 == 1:
            raise ValueError('Error: Please check whether kernel_size[0] % 2 == 1')
        temporal_window_size, max_graph_distance = kernel_size

        self.scn = Spatial_Basic_Block(in_channels, out_channels, max_graph_distance, A, **kwargs)
        self.tcn = Temporal_MultiScale_Block(out_channels, temporal_window_size, stride, **kwargs)
        if use_att:
            self.att = ST_Person_Attention(out_channels, **kwargs)
        else:
            self.att = lambda x: x
            

    def forward(self, x):
        return self.att(self.tcn(self.scn(x)))

class Input_Branch(nn.Module):
    def __init__(self, num_channel, A, use_att, **kwargs):
        super(Input_Branch, self).__init__()

        module_list = [
            Basic_Block(num_channel, 64, A, use_att, **kwargs),
            Basic_Block(64, 64, A, use_att, **kwargs),
            Basic_Block(64, 32, A, use_att, **kwargs)
        ]

        self.bn = nn.SyncBatchNorm(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V))
        for layer in self.layers:
            x = layer(x)

        return x

class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, A, edge_importance=True, adaptive=False, **kwargs):
        super(Spatial_Basic_Block, self).__init__()

        if in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.SyncBatchNorm(out_channels),
            ) 

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if adaptive:
            self.A = nn.Parameter(A[:max_graph_distance+1], requires_grad=True)
        else:
            self.register_buffer('A', A[:max_graph_distance+1])
        self.edge = nn.Parameter(torch.ones_like(A[:max_graph_distance+1]), requires_grad=edge_importance)


    def forward(self, x):

        res_block = self.residual(x)

        x = self.conv(x, self.A*self.edge)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x

class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, **kwargs):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.SyncBatchNorm(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.SyncBatchNorm(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x

class Temporal_MultiScale_Block(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, dilations=[1,2], residual_kernel_size=1, **kwargs):

        super().__init__()
        in_channels = out_channels
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.SyncBatchNorm(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        # pad = (3 + (3-1) * (1-1) - 1) // 2
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.SyncBatchNorm(branch_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(pad,0), stride=(stride,1)),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.SyncBatchNorm(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.SyncBatchNorm(branch_channels)
        ))

        # Residual connection
        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res 
        return out


class ST_Person_Attention(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias=True, **kwargs):
        super(ST_Person_Attention, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        self.mat = nn.Parameter(self.get_mean_matrix(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.SyncBatchNorm(inner_channel),
            nn.ReLU(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
    
        self.bn = nn.SyncBatchNorm(channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        P = len(self.parts)
        res = x

        x_t = x.mean(3, keepdims=True) # N,C,T,1
        # x_v = x.mean(2, keepdims=True).transpose(2, 3) # N,C,V,1
        x_p = (x.mean(2, keepdims=True) @ self.mat).transpose(2, 3) # N,C,P,1
        x_att = self.fcn(torch.cat([x_t, x_p], dim=2)) # N,C,(T+P),1
        x_t, x_p = torch.split(x_att, [T, P], dim=2) 
        x_t_att = self.conv_t(x_t).sigmoid() # N,C,T,1
        
        x_p_att = self.conv_v(x_p.transpose(2, 3)).sigmoid() # N,C,1,P
        x_v_att = x_p_att.index_select(3, self.joints) # N,C,1,V

        x_att = x_t_att * x_v_att
        return self.act(self.bn(x * x_att) + res)
    
    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)
    
    def get_mean_matrix(self):
        num_joints = sum([len(part) for part in self.parts])
        Q = torch.zeros(num_joints, len(self.parts))
        for j in range(len(self.parts)):
            n = len(self.parts[j])
            for joint in self.parts[j]:
                Q[joint][j] = 1.0/n
        return Q
    
    
# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x
    
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x