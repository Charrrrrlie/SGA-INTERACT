import torch
import torch.nn as nn
import torch.nn.functional as F


class Dynamic_Person_Inference(nn.Module):
    def __init__(self,
                 in_dim,
                 T,
                 N,
                 stride=1,
                 kernel_size=(3, 3),
                 dynamic_sampling=False,
                 sampling_ratio=[1],
                 group=1,
                 scale_factor=False,
                 beta_factor=False,
                 parallel_inference=False):
        super(Dynamic_Person_Inference, self).__init__()
        self.T = T
        self.N = N
        self.stride = stride
        self.kernel_size = kernel_size
        self.dynamic_sampling = dynamic_sampling
        self.sampling_ratio = sampling_ratio
        self.scale_factor = scale_factor
        self.max_ratio = sampling_ratio[-1]
        self.beta_factor = beta_factor
        self.parallel_inference = parallel_inference

        self.hidden_weight = nn.Linear(in_dim, in_dim, bias=False)

        if self.beta_factor:
            self.beta = nn.Parameter(torch.ones(len(sampling_ratio)), requires_grad=True)
            self.register_parameter('beta', self.beta)
        self.zero_padding = nn.ModuleDict()

        if self.dynamic_sampling:
            self.p_conv = nn.ModuleDict()
        if self.scale_factor:
            self.scale_conv = nn.ModuleDict()
        for ratio in self.sampling_ratio:
            pad_lr = (kernel_size[1] - 1)//2*ratio
            pad_tb = (kernel_size[0] - 1)//2*ratio
            self.zero_padding[str(ratio)] = nn.ZeroPad2d((pad_lr, pad_lr, pad_tb, pad_tb))

            if self.dynamic_sampling:
                ratio_p_conv = nn.Conv2d(in_channels = in_dim,
                                         out_channels = 2*kernel_size[0]*kernel_size[1],
                                         kernel_size = kernel_size,
                                         dilation = ratio,
                                         stride = stride,
                                         padding = (pad_tb, pad_lr),
                                         groups = group,
                                         bias = True)

                ratio_p_conv.weight.data.zero_()
                ratio_p_conv.bias.data.zero_()
                self.p_conv[str(ratio)] = ratio_p_conv

            if self.scale_factor:
                ratio_scale_conv = nn.Conv2d(in_channels = in_dim,
                                             out_channels = kernel_size[0]*kernel_size[1],
                                             kernel_size = kernel_size,
                                             dilation = ratio,
                                             stride = stride,
                                             padding = (pad_tb, pad_lr),
                                             groups = group,
                                             bias = True)

                ratio_scale_conv.weight.data.zero_()
                ratio_scale_conv.bias.data.zero_()
                self.scale_conv[str(ratio)] = ratio_scale_conv

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, person_features):
        '''

        :param person_features: shape [B, T, N, NFB]
        :return:
        '''
        person_features = person_features.permute(0, 3, 1, 2) # [B, NFB, T, N]
        ratio_feature = []
        for ratio in self.sampling_ratio:
            if self.parallel_inference:
                assert self.dynamic_sampling and self.scale_factor
                ratio_ft = self.parallel_infer(person_features, ratio)
                ratio_feature.append(ratio_ft)
            else:
                if self.dynamic_sampling:
                    ratio_ft = self.dynamic_infer_ratio(person_features, ratio)
                    ratio_feature.append(ratio_ft)
                else:
                    ratio_ft = self.plain_infer_ratio(person_features, ratio)
                    ratio_feature.append(ratio_ft)

        ratio_feature = torch.stack(ratio_feature, dim = 4)

        if self.beta_factor:
            dynamic_ft = torch.sum(self.beta * ratio_feature, dim = -1)
        else:
            dynamic_ft = torch.mean(ratio_feature, dim = 4)

        dynamic_ft = self.hidden_weight(dynamic_ft)

        return dynamic_ft


    def plain_infer_ratio(self, person_features, ratio):
        '''
        :param person_features: [B, NFB, T, N]
        :param ratio:
        :return:
        '''
        if self.scale_factor:
            scale = self.scale_conv[str(ratio)](person_features).permute(0, 2, 3, 1) # shape [B, T, N, k2]
            scale = F.softmax(scale, dim = -1)
        pos = self._get_plain_pos(ratio, person_features) # [B, T, N, 2*k2]

        pad_ft = self.zero_padding[str(ratio)](person_features).permute(0, 2, 3, 1) # [B, H, W, NFB]
        pad_ft = pad_ft.view(pad_ft.shape[0], -1, pad_ft.shape[-1])
        ft_pos = self._get_ft(pad_ft, pos.long(), ratio)

        if self.scale_factor:
            ft_infer = torch.sum(ft_pos * scale.unsqueeze(-1), dim = 3)
        else:
            ft_infer = torch.mean(ft_pos, dim = 3)

        return ft_infer


    def dynamic_infer_ratio(self, person_features, ratio):
        '''
        :param person_features: [B, NFB, T, N]
        :param ratio:
        :return:
        '''
        _, _, T, N = person_features.shape
        offset = self.p_conv[str(ratio)](person_features).permute(0, 2, 3, 1) # shape [B, T, N,  2*k2]

        # Dynamic relation matrix prediction using original features
        if self.scale_factor:
            scale = self.scale_conv[str(ratio)](person_features).permute(0, 2, 3, 1) # shape [B, T, N, k2]
            scale = F.softmax(scale, dim = -1)

        # Dynamic walk prediction using original features
        pos = self._get_pos(offset, ratio) # [B, T, N, 2*k2]

        # Original
        lt = pos.data.floor()
        rb = lt + 1

        # Calclate bilinear coefficient
        pad_lr = (self.kernel_size[1] - 1) // 2 * ratio
        pad_tb = (self.kernel_size[0] - 1) // 2 * ratio
        k2 = self.kernel_size[0]*self.kernel_size[1]
        lt = torch.cat((torch.clamp(lt[...,:k2], 0, T + 2 * pad_tb - 1),
                        torch.clamp(lt[...,k2:], 0, N + 2 * pad_lr - 1)), dim = -1)
        rb = torch.cat((torch.clamp(rb[...,:k2], 0, T + 2 * pad_tb - 1),
                        torch.clamp(rb[...,k2:], 0, N + 2 * pad_lr - 1)), dim = -1)
        lb = torch.cat((rb[...,:k2], lt[...,k2:]), dim = -1)
        rt = torch.cat((lt[...,:k2], rb[...,k2:]), dim = -1)

        # coefficient for cornor point pixel.  coe shape [B, T, N, k2]
        pos = torch.cat((torch.clamp(pos[...,:k2], 0, T + 2 * pad_tb - 1), # without -1
                         torch.clamp(pos[...,k2:], 0, N + 2 * pad_lr - 1)), dim = -1)
        coe_lt = (1 - torch.abs(pos[...,:k2] - lt[...,:k2]))*(1 - torch.abs(pos[...,k2:] - lt[...,k2:]))
        coe_rb = (1 - torch.abs(pos[...,:k2] - rb[...,:k2]))*(1 - torch.abs(pos[...,k2:] - rb[...,k2:]))
        coe_lb = (1 - torch.abs(pos[...,:k2] - lb[...,:k2]))*(1 - torch.abs(pos[...,k2:] - lb[...,k2:]))
        coe_rt = (1 - torch.abs(pos[...,:k2] - rt[...,:k2]))*(1 - torch.abs(pos[...,k2:] - rt[...,k2:]))

        # corner point feature.  ft shape [B, T, N, k2, NFB]
        pad_ft = self.zero_padding[str(ratio)](person_features).permute(0, 2, 3, 1)
        pad_ft = pad_ft.view(pad_ft.shape[0], -1, pad_ft.shape[-1])
        ft_lt = self._get_ft(pad_ft, lt.long(), ratio)
        ft_rb = self._get_ft(pad_ft, rb.long(), ratio)
        ft_lb = self._get_ft(pad_ft, lb.long(), ratio)
        ft_rt = self._get_ft(pad_ft, rt.long(), ratio)
        ft_infer = ft_lt * coe_lt.unsqueeze(-1) + \
                   ft_rb * coe_rb.unsqueeze(-1) + \
                   ft_lb * coe_lb.unsqueeze(-1) + \
                   ft_rt * coe_rt.unsqueeze(-1)

        if self.scale_factor:
            ft_infer =  torch.sum(ft_infer * scale.unsqueeze(-1), dim = 3)
        else:
            ft_infer = torch.mean(ft_infer, dim = 3)

        return ft_infer


    def parallel_infer(self, person_features, ratio):
        assert self.dynamic_sampling and self.scale_factor

        # Dynamic affinity infer
        scale = self.scale_conv[str(ratio)](person_features).permute(0, 2, 3, 1) # shape [B, T, N, k2]
        scale = F.softmax(scale, dim = -1)
        pos = self._get_plain_pos(ratio, person_features) # [B, T, N, 2*k2]

        pad_ft = self.zero_padding[str(ratio)](person_features).permute(0, 2, 3, 1) # [B, H, W, NFB]
        pad_ft = pad_ft.view(pad_ft.shape[0], -1, pad_ft.shape[-1])
        ft_pos = self._get_ft(pad_ft, pos.long(), ratio)

        ft_infer_scale =  torch.sum(ft_pos * scale.unsqueeze(-1), dim = 3)


        # Dynamic walk infer
        offset = self.p_conv[str(ratio)](person_features).permute(0, 2, 3, 1)  # shape [B, T, N,  2*k2]
        pos = self._get_pos(offset, ratio)  # [B, T, N, 2*k2]

        # Original
        lt = pos.data.floor()
        rb = lt + 1

        # Calclate bilinear coefficient
        # corner point position. lt shape # [B, T, N, 2*k2]
        k2 = self.kernel_size[0]*self.kernel_size[1]
        lt = torch.cat((torch.clamp(lt[..., :k2], 0, self.T + 2 * ratio - 1),
                        torch.clamp(lt[..., k2:], 0, self.N + 2 * ratio - 1)), dim=-1)
        rb = torch.cat((torch.clamp(rb[..., :k2], 0, self.T + 2 * ratio - 1),
                        torch.clamp(rb[..., k2:], 0, self.N + 2 * ratio - 1)), dim=-1)
        lb = torch.cat((rb[..., :k2], lt[..., k2:]), dim=-1)
        rt = torch.cat((lt[..., :k2], rb[..., k2:]), dim=-1)

        # coefficient for cornor point pixel.  coe shape [B, T, N, k2]
        pos = torch.cat((torch.clamp(pos[..., :k2], 0, self.T + 2 * ratio),
                         torch.clamp(pos[..., k2:], 0, self.N + 2 * ratio)), dim=-1)
        coe_lt = (1 - torch.abs(pos[..., :k2] - lt[..., :k2])) * (1 - torch.abs(pos[..., k2:] - lt[..., k2:]))
        coe_rb = (1 - torch.abs(pos[..., :k2] - rb[..., :k2])) * (1 - torch.abs(pos[..., k2:] - rb[..., k2:]))
        coe_lb = (1 - torch.abs(pos[..., :k2] - lb[..., :k2])) * (1 - torch.abs(pos[..., k2:] - lb[..., k2:]))
        coe_rt = (1 - torch.abs(pos[..., :k2] - rt[..., :k2])) * (1 - torch.abs(pos[..., k2:] - rt[..., k2:]))


        # corner point feature.  ft shape [B, T, N, k2, NFB]
        # pad_ft = self.zero_padding[ratio](person_features).permute(0, 2, 3, 1)
        # pad_ft = pad_ft.view(pad_ft.shape[0], -1, pad_ft.shape[-1])
        ft_lt = self._get_ft(pad_ft, lt.long(), ratio)
        ft_rb = self._get_ft(pad_ft, rb.long(), ratio)
        ft_lb = self._get_ft(pad_ft, lb.long(), ratio)
        ft_rt = self._get_ft(pad_ft, rt.long(), ratio)
        ft_infer_walk = ft_lt * coe_lt.unsqueeze(-1) + \
                   ft_rb * coe_rb.unsqueeze(-1) + \
                   ft_lb * coe_lb.unsqueeze(-1) + \
                   ft_rt * coe_rt.unsqueeze(-1)

        ft_infer_walk = torch.mean(ft_infer_walk, dim=3)

        return ft_infer_scale + ft_infer_walk


    def _get_ft(self, ft_map, idx, ratio):
        '''
        :param ft_map: shape [B, (T + 2*padding) * (N + 2*padding), NFB]
        :param idx: shape [B, T, N, 2*k2]
        :return:
        '''
        _, T, N, _ = idx.shape
        k2 = self.kernel_size[0]*self.kernel_size[1]
        padded_N = N + 2* ((self.kernel_size[1] - 1)//2*ratio)
        B, _, NFB = ft_map.shape

        reduced_idx = idx[...,:k2] * padded_N + idx[...,k2:]
        reduced_idx = reduced_idx.view(idx.shape[0], -1).contiguous()
        reduced_idx = reduced_idx.expand((NFB,) + reduced_idx.shape).permute(1,2,0)

        ft_idx = ft_map.gather(dim = 1, index = reduced_idx).view(B, T, N, k2, NFB)
        return ft_idx

    def _get_plain_pos(self, ratio, person_features):
        B, _, T, N =  person_features.shape
        device = person_features.device
        # meshgrid for kernel_size**2
        pos_k = self._get_pos_k(ratio, device)
        pos_0 = self._get_pos_0(T, N, ratio, device)
        plain_pos = (pos_0 + pos_k).repeat(B, 1, 1, 1)
        return plain_pos

    def _get_pos(self, offset, ratio):
        k2, T, N = offset.shape[3]//2, offset.shape[1], offset.shape[2] # k = kernel_size**2
        device = offset.device
        # meshgrid for kernel_size**2
        pos_k = self._get_pos_k(ratio, device)
        pos_0 = self._get_pos_0(T, N, ratio, device)

        return pos_0 + pos_k + offset

    def _get_pos_k(self, ratio, device):
        field_y = (self.kernel_size[0] - 1) * ratio + 1 # recptive field of a single conv
        field_x = (self.kernel_size[1] - 1) * ratio + 1  # recptive field of a single conv
        pos_k_y, pos_k_x = torch.meshgrid(torch.arange(-(field_y-1)//2, (field_y-1)//2+1, ratio, device = device),
                                          torch.arange(-(field_x-1)//2, (field_x-1)//2+1, ratio, device = device))
        pos_k = torch.cat((torch.flatten(pos_k_y), torch.flatten(pos_k_x)), dim = 0)
        pos_k = pos_k.view(1, 1, 1, 2*self.kernel_size[0]*self.kernel_size[1]).contiguous()
        return pos_k.float()

    def _get_pos_0(self, T, N, ratio, device):
        # Why torch.arange(1, h*self.stride+1, self.stride)
        pad_lr = (self.kernel_size[1] - 1) // 2 * ratio
        pad_tb = (self.kernel_size[0] - 1)//2*ratio
        pos_0_y, pos_0_x = torch.meshgrid(torch.arange(pad_tb, pad_tb + T*self.stride, self.stride, device = device),
                                          torch.arange(pad_lr, pad_lr + N*self.stride, self.stride, device = device))
        # Why need torch.flattent()?
        pos_0_y = pos_0_y.view(1, T, N, 1).expand(1, T, N, self.kernel_size[0]*self.kernel_size[1])
        pos_0_x = pos_0_x.view(1, T, N, 1).expand(1, T, N, self.kernel_size[0]*self.kernel_size[1])
        pos_0 = torch.cat((pos_0_y, pos_0_x), dim = 3)
        return pos_0.float()


class Multi_Dynamic_Inference(nn.Module):
    def __init__(self,
                 in_dim,
                 T,
                 N,
                 stride=1,
                 kernel_size=[(3, 3)],
                 dynamic_sampling=False,
                 sampling_ratio=[1],
                 group=1,
                 scale_factor=False,
                 beta_factor=False,
                 parallel_inference=False,
                 num_DIM=1):
        super(Multi_Dynamic_Inference, self).__init__()

        self.DIMlist = nn.ModuleList([Dynamic_Person_Inference(
                 in_dim=in_dim,
                 T=T,
                 N=N,
                 stride=stride,
                 kernel_size=kernel_size[i],
                 dynamic_sampling=dynamic_sampling,
                 sampling_ratio=sampling_ratio,
                 group=group,
                 scale_factor=scale_factor,
                 beta_factor=beta_factor,
                 parallel_inference=parallel_inference) for i in range(num_DIM)])

    def forward(self, person_features):
        DIM_features = []
        for i in range(len(self.DIMlist)):
            DIMft = self.DIMlist[i](person_features)
            DIM_features.append(DIMft)
        DIM_features = torch.sum(torch.stack(DIM_features, dim = 0), dim = 0)

        return DIM_features