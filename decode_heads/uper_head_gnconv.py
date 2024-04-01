# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from functools import partial
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch

import torch.nn as nn

from mmcv.utils import _BatchNorm, _InstanceNorm
from mmcv.cnn.utils import constant_init, kaiming_init
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.conv import build_conv_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.padding import build_padding_layer
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack, CONV_LAYERS
from scipy.spatial import distance
import numpy as np
from models.conv_custom import FrequencySelection
# @CONV_LAYERS.register_module('AdaDilatedConv')

class AdaptiveDilatedDWConv(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2
    def __init__(self, *args, 
                 offset_freq=None,
                 use_BFM=False,
                 kernel_decompose='both',
                 padding_mode='repeat',
                 normal_conv_dim=0,
                 pre_fs=True, # False, use dilation
                 fs_cfg={
                    # 'k_list':[3,5,7,9],
                    'k_list':[2,4,8],
                    'fs_feat':'feat',
                    'lowfreq_att':False,
                    # 'lp_type':'freq_eca',
                    # 'lp_type':'freq_channel_att',
                    # 'lp_type':'freq',
                    # 'lp_type':'avgpool',
                    'lp_type':'laplacian',
                    'act':'sigmoid',
                    'spatial':'conv',
                    # 'channel_res':True,
                    'spatial_group':1,
                },
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kernel_size[0] in (3, 7)
        assert self.groups == self.in_channels
        if kernel_decompose == 'both':
            self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
            self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'high':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'low':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1, groups=self.in_channels, reduction=0.0625, kernel_num=1, min_channel=16)
        self.kernel_decompose = kernel_decompose

        self.normal_conv_dim = normal_conv_dim

        if padding_mode == 'zero':
            self.PAD = nn.ZeroPad2d(self.kernel_size[0]//2)
        elif padding_mode == 'repeat':
            self.PAD = nn.ReplicationPad2d(self.kernel_size[0]//2)
        else:
            self.PAD = nn.Identity()
        print(self.in_channels, self.normal_conv_dim,)
        self.conv_offset = nn.Conv2d(
            self.in_channels - self.normal_conv_dim,
            self.deform_groups * 1,
            # self.groups * 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding if isinstance(self.PAD, nn.Identity) else 0,
            dilation=1,
            bias=True)
        # self.conv_offset_low = nn.Sequential(
        #     nn.AvgPool2d(
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         padding=1,
        #     ),
        #     nn.Conv2d(
        #         self.in_channels,
        #         self.deform_groups * 1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False),
        # )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(
                self.in_channels - self.normal_conv_dim,
                self.in_channels - self.normal_conv_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding if isinstance(self.PAD, nn.Identity) else 0,
                groups=self.in_channels - self.normal_conv_dim,
                dilation=1,
                bias=False),
            nn.Conv2d(
                self.in_channels - self.normal_conv_dim,
                self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                dilation=1,
                bias=True)
        )
        
        self.offset_freq = offset_freq

        if self.offset_freq in ('FLC_high', 'FLC_res'):
            self.LP = FLC_Pooling(freq_thres=min(0.5 * 1 / self.dilation[0], 0.25))
        elif self.offset_freq in ('SLP_high', 'SLP_res'):
            self.LP = StaticLP(self.in_channels, kernel_size=5, stride=1, padding=2, alpha=8)
        elif self.offset_freq is None:
            pass
        else:
            raise NotImplementedError

        # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
        if self.kernel_size[0] == 3:
            offset = [-1, -1,  -1, 0,   -1, 1,
                    0, -1,   0, 0,    0, 1,
                    1, -1,   1, 0,    1,1]
        elif self.kernel_size[0] == 7:
            offset = [
                -3, -3,  -3, -2,  -3, -1,  -3, 0,  -3, 1,  -3, 2,  -3, 3, 
                -2, -3,  -2, -2,  -2, -1,  -2, 0,  -2, 1,  -2, 2,  -2, 3, 
                -1, -3,  -1, -2,  -1, -1,  -1, 0,  -1, 1,  -1, 2,  -1, 3, 
                0, -3,   0, -2,   0, -1,   0, 0,   0, 1,   0, 2,   0, 3, 
                1, -3,   1, -2,   1, -1,   1, 0,   1, 1,   1, 2,   1, 3, 
                2, -3,   2, -2,   2, -1,   2, 0,   2, 1,   2, 2,   2, 3, 
                3, -3,   3, -2,   3, -1,   3, 0,   3, 1,   3, 2,   3, 3, 
            ]
        else: raise NotImplementedError

        offset = torch.Tensor(offset)
        # offset[0::2] *= self.dilation[0]
        # offset[1::2] *= self.dilation[1]
        # a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
        self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None])) # B, G, 49, 1, 1
        self.init_weights()

        self.use_BFM = use_BFM
        if use_BFM:
            alpha = 8
            BFM = np.zeros((self.in_channels, 1, self.kernel_size[0], self.kernel_size[0]))
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[0]):
                    point_1 = (i, j)
                    point_2 = (self.kernel_size[0]//2, self.kernel_size[0]//2)
                    dist = distance.euclidean(point_1, point_2)
                    BFM[:, :, i, j] = alpha / (dist + alpha)
            self.register_buffer('BFM', torch.Tensor(BFM))
            print(self.BFM)
        if fs_cfg is not None:
            if pre_fs:
                self.FS = FrequencySelection(self.in_channels - self.normal_conv_dim, **fs_cfg)
            else:
                self.FS = FrequencySelection(1, **fs_cfg) # use dilation
        self.pre_fs = pre_fs

    def freq_select(self, x):
        if self.offset_freq is None:
            pass
        elif self.offset_freq in ('FLC_high', 'SLP_high'):
            x - self.LP(x)
        elif self.offset_freq in ('FLC_res', 'SLP_res'):
            2 * x - self.LP(x)
        else:
            raise NotImplementedError
        return x

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.fill_((self.dilation[0] - 1)/self.dilation[0])
            # self.conv_offset.bias.data.zero_()
        # if hasattr(self, 'conv_offset_low'):
            # self.conv_offset_low[1].weight.data.zero_()
        if hasattr(self, 'conv_mask'):
            self.conv_mask[1].weight.data.zero_()
            self.conv_mask[1].bias.data.zero_()

    def forward(self, x):
        if self.normal_conv_dim > 0:
            return self.mix_forward(x)
        else:
            return self.ad_forward(x)
        
    def ad_forward(self, x):
        if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            c_att1, _, _, _, = self.OMNI_ATT1(x)
            c_att2, _, _, _, = self.OMNI_ATT2(x)
        elif hasattr(self, 'OMNI_ATT'):
            c_att, _, _, _, = self.OMNI_ATT(x)
        x = self.PAD(x)
        offset = self.conv_offset(x)
        offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
        if hasattr(self, 'FS') and (self.pre_fs==False): x = self.FS(x, offset)
        b, _, h, w = offset.shape
        offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
        offset = offset.reshape(b, -1, h, w)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            offset = offset.reshape(1, -1, h, w)
            # print(offset.max(), offset.min(), offset.mean())
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, out, in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            adaptive_weight = adaptive_weight_mean * (2 * c_att1.unsqueeze(2)) + (adaptive_weight - adaptive_weight_mean) * (2 * c_att2.unsqueeze(2))
            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        (1, 1), # dilation
                                        self.groups * b, self.deform_groups * b)
            return x.reshape(b, -1, h, w)
        elif hasattr(self, 'OMNI_ATT'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, out, in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            if self.kernel_decompose == 'high':
                adaptive_weight = adaptive_weight_mean + (adaptive_weight - adaptive_weight_mean) *  (2 * c_att.unsqueeze(2))
            elif self.kernel_decompose == 'low':
                adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(2)) + (adaptive_weight - adaptive_weight_mean) 
            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        (1, 1), # dilation
                                        self.groups * b, self.deform_groups * b)
            return x.reshape(b, -1, h, w)
        else:
            return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        self.dilation, self.groups,
                                        self.deform_groups)
    def mix_forward(self, x):
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            c_att1, _, _, _, = self.OMNI_ATT1(x)
            c_att2, _, _, _, = self.OMNI_ATT2(x)
        elif hasattr(self, 'OMNI_ATT'):
            c_att, _, _, _, = self.OMNI_ATT(x)
        ori_x = x
        normal_conv_x = ori_x[:, -self.normal_conv_dim:] # ad:normal
        x = ori_x[:, :-self.normal_conv_dim]
        if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
        x = self.PAD(x)
        offset = self.conv_offset(x)
        if hasattr(self, 'FS') and (self.pre_fs==False): x = self.FS(x, offset)
        offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
        b, _, h, w = offset.shape
        offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
        offset = offset.reshape(b, -1, h, w)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            offset = offset.reshape(1, -1, h, w)
            # print(offset.max(), offset.min(), offset.mean())
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, out, in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            adaptive_weight = adaptive_weight_mean * (2 * c_att1.unsqueeze(2)) + (adaptive_weight - adaptive_weight_mean) * (2 * c_att2.unsqueeze(2))
            # adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight[:, :-self.normal_conv_dim].reshape(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]), self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        (1, 1), # dilation
                                        (self.in_channels - self.normal_conv_dim) * b, self.deform_groups * b)
            x = x.reshape(b, -1, h, w)
            normal_conv_x = F.conv2d(normal_conv_x.reshape(1, -1, h, w), adaptive_weight[:, -self.normal_conv_dim:].reshape(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]), 
                                     bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.normal_conv_dim * b)
            normal_conv_x = normal_conv_x.reshape(b, -1, h, w)
            # return torch.cat([normal_conv_x, x], dim=1)
            return torch.cat([x, normal_conv_x], dim=1)
        elif hasattr(self, 'OMNI_ATT'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, out, in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            if self.kernel_decompose == 'high':
                adaptive_weight = adaptive_weight_mean + (adaptive_weight - adaptive_weight_mean) *  (2 * c_att.unsqueeze(2))
            elif self.kernel_decompose == 'low':
                adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(2)) + (adaptive_weight - adaptive_weight_mean) 
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight[:, :-self.normal_conv_dim].reshape(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]), self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        (1, 1), # dilation
                                        (self.in_channels - self.normal_conv_dim) * b, self.deform_groups * b)
            x = x.reshape(b, -1, h, w)
            normal_conv_x = F.conv2d(normal_conv_x.reshape(1, -1, h, w), adaptive_weight[:, -self.normal_conv_dim:].reshape(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]), 
                                     bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.normal_conv_dim * b)
            normal_conv_x = normal_conv_x.reshape(b, -1, h, w)
            # return torch.cat([normal_conv_x, x], dim=1)
            return torch.cat([x, normal_conv_x], dim=1)
        else:
            return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                        self.stride, self.padding if isinstance(self.PAD, nn.Identity) else 0, #padding
                                        self.dilation, self.groups,
                                        self.deform_groups)
        # print(x.shape)
        
def get_dwconv(dim, kernel, dilation, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 * dilation, dilation=dilation, bias=bias, groups=dim)

def get_adwconv(dim, kernel, dilation, bias):
    return AdaptiveDilatedDWConv(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 * dilation, dilation=dilation, bias=bias, groups=dim)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
    

class GlobalLocalConv2Norm(nn.Module):
    def __init__(self, dim, h=14, w=8, kernel_size=3, use_adaptive_dilation=True):
        super().__init__()
        if use_adaptive_dilation:
            self.dw = AdaptiveDilatedDWConv(dim // 2, dim // 2, kernel_size=kernel_size, padding=1, bias=False, groups=dim // 2)
            # self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=kernel_size, padding=1, bias=False, groups=32)
            # self.dw = AdaptiveDilatedConv(dim // 2, dim // 2, kernel_size=kernel_size, padding=1, bias=False, groups=1)
        else:
            self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=kernel_size, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


# @PLUGIN_LAYERS.register_module()
class GNConvModule(nn.Module):
    def __init__(self, 
        in_channels,
        out_channels,
        use_adaptive_dilation=True,
        kernel_size=3,
        dilation=1,
        dim_ratio=1,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        inplace=False,
        proj_out=False,
        order=2,
        type='gnconv',
        h=32,
        gf_kernel_size=3,
    ):
        super().__init__()
        w = h // 2 + 1
        self.order = order
        dim = out_channels * dim_ratio
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        print(f'[gconv]: kernel size = {kernel_size}')
        print(f'[gconv]: dim ratio = {dim_ratio}')
        print('[gconv]', order, 'order with dims=', self.dims)
        self.proj_in = nn.Conv2d(in_channels, 2 * dim, 1)

        if type == 'gnconv':
            if use_adaptive_dilation:
                self.dwconv = get_adwconv(sum(self.dims), kernel_size, dilation, True) # nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=3, groups=sum(self.dims))
            else:
                self.dwconv = get_dwconv(sum(self.dims), kernel_size, dilation, True) # nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=3, groups=sum(self.dims))
        elif type == 'gngf':
            self.dwconv = GlobalLocalConv2Norm(sum(self.dims), h=h, w=w, kernel_size=gf_kernel_size, use_adaptive_dilation=use_adaptive_dilation) # nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=3, groups=sum(self.dims))
        else:
            raise NotImplementedError()

        if proj_out:
            self.proj_out = nn.Conv2d(dim, out_channels, 1)
        else:
            self.proj_out = nn.Identity()

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # build normalization layers
        if self.with_norm:
            norm_channels = out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc)

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)

        return x


@HEADS.register_module()
class UPerHead_Horad(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), 
            conv_cfg=dict(
                kernel_size=3,
                dim_ratio=1,
                type='gnconv', 
                proj_out=False,
                order=2,
                use_adaptive_dilation=True,
            ),
            btn_kernel=3,
            **kwargs):
        super().__init__(
            input_transform='multiple_select', **kwargs)
        
        GNModule = partial(GNConvModule, **conv_cfg)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                # conv_cfg=self.conv_cfg,
                conv_cfg={
                    'type':'AdaDilatedConv', 'padding_mode':'repeat', 'kernel_decompose':'high', 
                    'pre_fs':False,
                    'conv_type':'conv',
                    'fs_cfg':{
                        # 'k_list':[3,5,7,9],
                        'k_list':[2,4,8],
                        'fs_feat':'feat',
                        'lowfreq_att':False,
                        # 'lp_type':'freq_eca',
                        # 'lp_type':'freq_channel_att',
                        # 'lp_type':'freq',
                        # 'lp_type':'avgpool',
                        'lp_type':'laplacian',
                        'act':'sigmoid',
                        'spatial':'conv',
                        'channel_res':True,
                        'spatial_group':1,
                    },}, ###
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = GNModule(
            len(self.in_channels) * self.channels,
            self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    

if __name__ == '__main__':
    fpn_conv = GNConvModule(
        384,
        256,
        3,
        norm_cfg = dict(type='SyncBN', requires_grad=True),
        inplace=False)

    x = torch.randn(10, 384, 224, 224)
    x = fpn_conv(x)
    print(x.shape)
