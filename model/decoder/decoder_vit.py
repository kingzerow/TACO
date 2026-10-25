import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.GELU, nn.ReLU6, nn.InstanceNorm2d, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.UpsamplingBilinear2d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 4)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)

class Deep_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Deep_Block, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 4)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)

class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)

########################################### CoordAttention #########################################
# Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class SFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFA, self).__init__()
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.css = CrossStrengthen(out_channel)

    def forward(self, x, y):

        N, C, H, W = x.size()
        x0 = F.interpolate(self.branch0(x), H)
        x1 = F.interpolate(self.branch1(x), H)
        x2 = F.interpolate(self.branch2(x), H)
        if x.size() != y.size():
            y = F.interpolate(y, H)
        out = self.css(x0, y)
        out = self.css(out, x1)
        out = self.css(out, x2)
        return out

    def initialize(self):
        weight_init(self)

class MaskAttention(nn.Module):
    def __init__(self, channel):
        super(MaskAttention, self).__init__()
        LayerNorm_type = 'WithBias'
        bias = False
        ffn_expansion_factor = 4
        num_heads = 8
        mode = 'dilation'
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.norm1 = LayerNorm(channel, LayerNorm_type)
        self.attn = Attention(channel, num_heads, bias, mode)
        self.norm2 = LayerNorm(channel, LayerNorm_type)
        self.ffn = FeedForward(channel, ffn_expansion_factor, bias)
        self.fuse = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1), nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

    def forward(self, x, y):
        x = x * y
        out = F.relu(self.bn_1(self.conv_1(x)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        out = out + self.attn(self.norm1(out))
        out = out + self.ffn(self.norm2(out))
        out = self.fuse(x + x * out)
        return out

    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape


    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class SFTA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFTA, self).__init__()
        self.down_channel = nn.Sequential(
            conv3x3(in_channel, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            conv3x3(out_channel * 2, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.TA = MSA_head(dim=out_channel)
    def forward(self, x, y):
        _, _, H1, _ = x.size()
        _, _, H2, _ = y.size()
        if H1 != H2:
            y = self.up(y)
        x = self.down_channel(x)
        out = self.fuse(torch.cat((x, y), dim=1))
        out = self.TA(out)
        return out

    def initialize(self):
        weight_init(self)
class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class CrossStrengthen(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(CrossStrengthen, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature3 = nn.Parameter(torch.ones(num_heads, 1, 1))
        ffn_expansion_factor = 4
        # x
        self.qkv_x_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_x_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_x_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkvx1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkvx2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)
        self.qkvx3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)

        # y
        self.qkv_y_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_y_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_y_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkvy1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=bias)
        self.qkvy2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkvy3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.norm_x = LayerNorm(dim, LayerNorm_type)
        self.norm_y = LayerNorm(dim, LayerNorm_type)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x, y):
        input = x
        # Attention part
        B, C, H, W = x.shape
        x = self.norm_x(x)
        y = self.norm_y(y)
        qx = self.qkvx1conv(self.qkv_x_0(x))
        kx = self.qkvx2conv(self.qkv_x_1(x))
        vx = self.qkvx3conv(self.qkv_x_2(x))
        qx = rearrange(qx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qx = torch.nn.functional.normalize(qx, dim=-1)
        kx = torch.nn.functional.normalize(kx, dim=-1)

        qy = self.qkvy1conv(self.qkv_y_0(y))
        ky = self.qkvy2conv(self.qkv_y_1(y))
        vy = self.qkvy3conv(self.qkv_y_2(y))
        qy = rearrange(qy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ky = rearrange(ky, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vy = rearrange(vy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qy = torch.nn.functional.normalize(qy, dim=-1)
        ky = torch.nn.functional.normalize(ky, dim=-1)

        attnx = ((qx @ ky.transpose(-2, -1)) * self.temperature1).softmax(dim=-1)
        attny = ((qy @ kx.transpose(-2, -1)) * self.temperature2).softmax(dim=-1)

        out = (attnx @ attny @ vx) @ (vx.transpose(-2, -1) @ vy) * self.temperature3
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.project_out(out)

        out = input + out
        out = out + self.ffn(self.norm(out))
        out = self.fuse(input + input * out)
        return out
    def initialize(self):
        weight_init(self)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.pyramid_pooling = PyramidPooling(512, channels)
        self.sfa1 = SFTA(512, channels)
        self.sfa2 = SFTA(320, channels)
        self.sft1 = SFTA(128, channels)
        self.sft2 = SFTA(64, channels)
        self.out1 = OutPut(in_chs=channels, scale=32)
        self.out2 = OutPut(in_chs=channels, scale=32)
        self.out3 = OutPut(in_chs=channels, scale=16)
        self.out4 = OutPut(in_chs=channels, scale=8)
        self.out5 = OutPut(in_chs=channels, scale=4)
        self.initialize()
    def forward(self, E1, E2, E3, E4, shape):
        # E1 512 12 E2 320 24 E3 128 48 E4 64 96 96
        SM = self.pyramid_pooling(E1)
        S4 = self.sfa1(E1, SM)
        S3 = self.sfa2(E2, S4)
        S2 = self.sft1(E3, S3)
        S1 = self.sft2(E4, S2)
        P5 = self.out1(SM)
        P4 = self.out2(S4)
        P3 = self.out3(S3)
        P2 = self.out4(S2)
        P1 = self.out5(S1)
        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)
