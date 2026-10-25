import torch
import torch.nn as nn
import torch.nn.functional as F
# import backbone.resnet as resnet
from .encoder.pvtv2 import pvt_v2_b4, pvt_v2_b2
# from lib.model.mobilemamba import MobileMamba_B2
from model.modules.mmfusion import MultiModalFusion, MultiModalAttention
# from model.modules.mmfusion1 import MultiModalFusion, MultiModalAttention
from math import log
import numpy as np
from torch.nn import init
import einops
from einops import rearrange
import numbers
import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/user1/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

    def initialize(self):
        pass


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        output = self.sigmoid(max_out + avg_out)  # 相加,然后通过sigmoid获得权重:(B,C,1,1)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W)
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel, reduction=reduction)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # residual=x
        out = x * self.ChannelAttention(x)  # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        out = out * self.SpatialAttention(out)  # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        # return out+residual
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):  # act = ReLU or PReLU
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class RFD(nn.Module):
    def __init__(self, channel, output, kernel_size, reduction, bias, act, n_resblocks):
        super(RFD, self).__init__()
        modules_body = [RCAB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.conv = Conv1x1(channel, output)

    def forward(self, x):
        res = self.body(x)
        res += x
        res = self.conv(res)
        return res


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.PReLU()
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class diff_moudel(nn.Module):
    def __init__(self, in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()

    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # weight = self.conv_1(edge)
        out = weight * x + x
        out = self.simam(out)
        return out


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        # self.reduce1 = Conv1x1(64, 16)
        # self.reduce4 = Conv1x1(64, 64)
        self.diff_1 = diff_moudel(64)
        self.diff_2 = diff_moudel(64)
        self.cbam = CBAMBlock(channel=128)
        # self.ema = EMA(channels=80)
        self.block = nn.Sequential(
            ConvBNR(128, 64, 3),
            ConvBNR(64, 64, 3),
            nn.Conv2d(64, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        # x1 = self.reduce1(x1)
        # x4 = self.reduce4(x4)
        x1 = self.diff_1(x1)
        x4 = self.diff_2(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.cbam(out)
        # fb = out
        out = self.block(out)

        # return out, fb
        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        # t = int(abs((log(channel, 2) + 1) / 2))
        # k = t if t % 2 else t + 1
        # self.conv2d = ConvBNR(channel, channel, 3)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # self.cbam = CBAMBlock(channel=channel)
        # self.sigmoid = nn.Sigmoid()
        # self.hor = HOR()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        # x = self.conv2d(x)
        # wei = self.avg_pool(x)
        # wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # wei = self.sigmoid(wei)
        # x = x * wei
        # if i == 1 or i == 2:
        #     x = x + self.cbam(x)
        # elif i == 3 or i == 4:
        #     x = x + self.hor(x)

        return x


class Att(nn.Module):
    def __init__(self, channels=64, r=4):
        super(Att, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.alpha = nn.Parameter(torch.ones(1))
        self.sig = nn.Sigmoid()
        self.conv_l = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_g = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1 = self.conv_l(x)
        x2 = self.conv_g(x)
        xl = self.local_att(x)
        xg = self.global_att(x)
        # xl = self.local_att(x1+x2)
        # xg = self.global_att(x1+x2)
        xlg = xl + xg
        wei = self.sig(xlg)
        xo = self.alpha * x1 * wei + self.alpha * x2 * (1 - wei)
        # print(self.alpha)
        # xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        return x + xo

    def initialize(self):
        # weight_init(self)
        pass


class HOR(nn.Module):
    def __init__(self, channel=64):
        super(HOR, self).__init__()
        self.high = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.low = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

        self.value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.e_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.mid = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.latter = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())
        # self.fac = nn.Parameter(torch.ones(1))
        # self.fac1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_latter = self.conv(x)
        b, c, h, w = x_latter.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(x_latter).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()
        # print('x_', x_.shape, 'x_latter_', x_latter_.shape)
        p = torch.bmm(x_, x_latter_).contiguous()
        p = self.softmax(p).contiguous()

        # print('p', p.shape)
        e_ = torch.bmm(p, self.value(x).reshape(b, c, h * w).permute(0, 2, 1)).contiguous()
        # print('e_', e_.shape)
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        e = self.e_conv(e.reshape(b, c, h, w)).reshape(b, c, h * w).contiguous()

        # e = e.permute(0, 2, 1)
        x_latter_ = self.latter(x_latter).reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        # print('e', e.shape, 'x_latter_', x_latter_.shape)
        t = torch.bmm(e, x_latter_).contiguous()
        t = self.softmax(t).contiguous()

        x_ = self.mid(x).view(b, c_, h * w).permute(0, 2, 1).contiguous()
        # print('x_', x_.shape, 't', t.shape)
        out = torch.bmm(x_, t).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return out


class LocalAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(dim, num_heads=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, self.window_size * self.window_size).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)  # [16, 2304, 384])
        attn_output = attn_output.permute(1, 2, 0).view(B, H // self.window_size, W // self.window_size, C,
                                                        self.window_size, self.window_size)  # [1, 48, 48, 384, 4, 4])
        attn_output = attn_output.permute(0, 3, 1, 4, 2, 5).contiguous()  # ([1, 384, 48, 4, 48, 4])
        attn_output = attn_output.view(B, C, H, W)  # [1, 384, 192, 192]
        return attn_output


class ImprovedModule(nn.Module):
    def __init__(self, dim):
        super(ImprovedModule, self).__init__()
        self.local_attn = LocalAttention(dim, window_size=4)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.local_attn(x)
        x = self.relu(self.conv(x))
        return x


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):  # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):  # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class FIEM(nn.Module):
    def __init__(self, dim_in=32, dim_mid=16, mids=4, img_size=384, up=False):
        """
        dim_in : input channel
        dim_mid ：
        """
        super(FIEM, self).__init__()

        self.normalize = nn.LayerNorm(dim_in)
        self.img_size = img_size
        self.num_s = int(dim_mid)
        self.num_n = (mids * mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, dim_in, kernel_size=1, bias=False)

        if up:
            self.local_attn = ImprovedModule(dim_in)
        else:
            self.local_attn = nn.Identity()

    def forward(self, x1, x2):
        n, c, h, w = x1.size()  # torch.Size([8, 64, 104, 104])
        x2 = self.local_attn(x2)  # torch.Size([8, 1, 13, 13])
        x2 = torch.nn.functional.softmax(x2, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x1).view(n, self.num_s, -1)  # 1x1conv downsampling torch.Size([8, 16, 9216])
        x_proj = self.conv_proj(x1)# torch.Size([8, 16, 96, 96])
        x_mask = x_proj * x2 # torch.Size([8, 16, 96, 96])

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1) # torch.Size([8, 16, 16])

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1)) # 
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x2.size()[2:])
        out = x1 + (self.conv_extend(x_state))

        return out

class TextProcessor(nn.Module):
    """
    Module for processing referring text data
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(TextProcessor, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, in_text_dict):
        # Extract from the input dict
        # encoded_text = in_text_dict['encoded_text']
        text_features = in_text_dict['text_features'].squeeze(1)
        text_attention_mask = in_text_dict['text_attention_mask'].squeeze(1)
        
        # Perform projection on text features
        text_features = self.proj(text_features)
        text_features = self.layer_norm(text_features)
        text_features = self.dropout(text_features)
        text_masks = text_attention_mask
        text_features = NestedTensor(text_features, text_masks)
        
        # Post-process the text features
        text_word_features, text_word_masks = text_features.decompose()
        # # except embedded shape: [B, LEN, C]
        # embedded = text_word_features * text_word_masks.unsqueeze(-1)
        # # except text_sentence_features shape: [B, C]
        # text_sentence_features = embedded.sum(1) / (text_word_masks.ne(1).sum(-1).unsqueeze(-1).float())

        valid_mask = text_word_masks.ne(1).unsqueeze(-1).float()
        embedded = text_word_features * valid_mask
        text_sentence_features = embedded.sum(1) / (valid_mask.sum(1) + 1e-6)

        feature_dict = {
            "refs": text_word_features,             # [B, SEQ_LEN, C]
            # "ref_values": text_word_features,     # [B, SEQ_LEN, C]
            "masks": text_word_masks,               # [B, SEQ_LEN]
            "ref_embeds": text_sentence_features,   # [B, C]
        }
        return feature_dict

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
class MyNet(nn.Module):
    def __init__(self, encoder='pvt_v2_b4', channel=64, clip_img_size=True, text_encoders='clip-large'):
        super(MyNet, self).__init__()

        # backbone 的加载
        # self.backbone = pvt_v2_b2()
        # path = './pretrained_pvt/pvt_v2_b2.pth'
        if encoder == 'pvt_v2_b2':
            self.encoder = pvt_v2_b2()
        elif encoder == 'pvt_v2_b4':
            self.encoder = pvt_v2_b4()
        self.clip_img_size = clip_img_size
        if self.clip_img_size:
            self.text_encoders = text_encoders
            self.prepare_vision_encoder()
            # self.clip_vision_proj =nn.Conv2d(768, 64, 1, 1, 0)
            if text_encoders == 'clip-base':
                self.clip_vision_proj =nn.Conv2d(768, 64, 1, 1, 0)
                text_encoders_out_dim = 512
            elif text_encoders == 'clip-large':
                text_encoders_out_dim = 768
                self.clip_vision_proj =nn.Conv2d(1024, 64, 1, 1, 0)
            self.text_processor = TextProcessor(
                in_dim=text_encoders_out_dim,
                out_dim=64
            )
        self.channels = [512, 320, 128, 64]
        self.cr4 = nn.Sequential(nn.Conv2d(512, channel, 1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(320, channel, 1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(128, channel, 1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(64, channel, 1), nn.BatchNorm2d(channel), nn.PReLU())

        # self.conv1 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], channel),
        #                            nn.BatchNorm2d(channel), nn.PReLU())
        # self.conv2 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], channel),
        #                            nn.BatchNorm2d(channel), nn.PReLU())
        # self.conv3 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], channel),
        #                            nn.BatchNorm2d(channel), nn.PReLU())
        # self.conv4 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], channel),
        #                            nn.BatchNorm2d(channel), nn.PReLU())
        # self.conv5 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())
        # self.conv6 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())

        # self.cbr1 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())

        # self.Att1 = Att()
        # self.Att2 = Att()
        # self.Att3 = Att()
        self.sam1 = FIEM(channel)
        self.sam2 = FIEM(channel)
        # self.cross_modal_attention = MultiModalAttention(
        #     query_dim=64,
        #     key_dim=64,
        #     value_dim=64,
        #     head_dim=64,
        #     num_heads=8,
        #     dropout=0.1
        # )
        # self.eam = EAM()
        self.efm1 = EFM(channel)
        self.efm2 = EFM(channel)
        self.efm3 = EFM(channel)
        self.efm4 = EFM(channel)
        # self.hor3 = HOR()
        # self.hor4 = HOR()
        # self.cbr1 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        # self.cbr2 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        # self.cbr3 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        # self.cbr4 = nn.Sequential(nn.Conv2d(64,64,3,1,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, 3, 1, 1, 1),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            ) for _ in range(4)  # 数量对应原cbr1~cbr4
        ])
        self.fushion_layer_list = nn.ModuleList([
            MultiModalFusion(
                img_dim=channel,
                text_dim=channel
            ) for _ in range(4)  # 数量对应原cbr1~cbr4
        ])

        self.map1 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.map2 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.map3 = nn.Conv2d(channel, 1, 3, 1, 1)
        # self.out_map = nn.Conv2d(channel, 1, 7, 1, 3)
        self.out1 = nn.Sigmoid()
        self.out2 = nn.Sigmoid()
        self.out3 = nn.Sigmoid()

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)
    
    def prepare_vision_encoder(self):
        from transformers import CLIPVisionModel
        if self.text_encoders == 'clip-base':
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif self.text_encoders == 'clip-large':
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
            # self.vision_encoder = CLIPVisionModel.from_pretrained("../wyq/pretrained/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError(f"Text encoder {self.config.text_encoders} not supported.")
        assert self.vision_encoder is not None, f"Vision encoder {self.config.vision_encoders} not found."
        
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)
        self.vision_encoder = self.vision_encoder.to('cuda')

    def denormalize(self, tensor, imgsize=336, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    
        tensor = tensor * std + mean
        tensor = F.interpolate(tensor, size=imgsize, mode='bilinear', align_corners=True)
        tensor = (tensor - mean) / std
        return tensor
    
    def forward(self, x, text_feature, enable_ref_text=True):
        """
            x : 3  * 448 * 448
            x1: 64 * 176 * 176
            x2: 128* 88  * 88
            x3: 320* 44  * 44
            x4: 512* 22  * 22

        """

        # 提取特征
        pvt = self.encoder(x)
        x1 = pvt[0]  # torch.Size([4, 64, 176, 176])
        x2 = pvt[1]  # torch.Size([4, 128, 88, 88])
        x3 = pvt[2]  # torch.Size([4, 320, 44, 44])
        x4 = pvt[3]  # torch.Size([4, 512, 22, 22])

        x4 = self.cr4(x4)  # 64
        x3 = self.cr3(x3)  # 64
        x2 = self.cr2(x2)  # 64
        x1 = self.cr1(x1)  # 64

        gt_x4 = x1
        edge_x1 = self.sam1(x1, F.interpolate(x4, size=x1.size()[2:], mode='bilinear'))
        gt_x4 = self.sam2(F.interpolate(x4, size=x1.size()[2:], mode='bilinear'), x1)

        # edge_x1 = torch.sigmoid(edge_x1)
        # gt_x4 = torch.sigmoid(gt_x4)
        x1 = self.efm1(x1, torch.sigmoid(edge_x1))
        x2 = self.efm2(x2, torch.sigmoid(edge_x1))
        x3 = self.efm3(x3, torch.sigmoid(gt_x4))
        x4 = self.efm4(x4, torch.sigmoid(gt_x4))  # x4:torch.Size([4, 512, 22, 22])


        x1_proj = x1  # 64
        x2_proj = x2  # 64
        x3_proj = x3  # 64
        x4_proj = x4  # 64

        if enable_ref_text:
            assert text_feature is not None, "Referring text dict is not provided!"
            # if self.enable_text_aug:
            #     text_feature, text_loss = self.text_aug(text_feature, x4.detach(), hash_labels, gt)
            text_feature = self.text_processor(text_feature)
            # x1_proj, x2_proj, x3_proj, x4_proj = [self.in_img_proj[i](x) for i, x in enumerate([x1, x2, x3, x4])]
            if self.clip_img_size:
                vision_features = self.denormalize(x)
                vision_features = self.vision_encoder(vision_features).last_hidden_state
                vision_features = vision_features[:,1:,:].reshape(vision_features.shape[0], int(vision_features.shape[1]**0.5), int(vision_features.shape[1]**0.5), vision_features.shape[2]).permute(0, 3, 1, 2)
                vision_features = self.clip_vision_proj(vision_features)
                x1_proj, x2_proj, x3_proj, x4_proj = (x_proj + F.interpolate(vision_features, size=x_proj.shape[2:], mode='bilinear', align_corners=True) for x_proj in (x1_proj, x2_proj, x3_proj, x4_proj))
            proj = [x4_proj, x3_proj, x2_proj, x1_proj]
            for i in range(4):
                # 1. 融合当前层的图像和文本
                # 注意：这里我们直接处理 proj[i]
                fused_out = self.fushion_layer_list[i](proj[i], text_feature)[0]

                # 2. 如果不是最后一层(x1)，则将结果上采样并传递给下一层
                if i < 3:
                    # 这里的 size 取自 proj[i+1]，即更浅一层的特征图尺寸
                    t = F.interpolate(self.cbr_list[i](fused_out), size=proj[i+1].size()[2:], mode='bilinear')
                    # 残差连接：将高层语义注入到下一层
                    proj[i+1] = proj[i+1] + t
                else:
                    # 3. 如果是最后一层 (i == 3, 即 x1)，处理完后直接赋值
                    proj[i] = self.cbr_list[i](fused_out)
        rf1 = proj[3]
        edge_x1 = self.map1(edge_x1)
        gt_x4 = self.map2(gt_x4)
        rf1_map = self.map3(rf1)

        # gf_pre = F.interpolate(gf_pre, size=x.size()[2:], mode='bilinear')  # torch.Size([4, 1, 704, 704])
        rf1_map = F.interpolate(rf1_map, size=x.size()[2:], mode='bilinear')
        edge1 = F.interpolate(edge_x1, size=x.size()[2:], mode='bilinear', align_corners=False)
        gt4 = F.interpolate(gt_x4, size=x.size()[2:], mode='bilinear', align_corners=False)
        # gf_pre = self.out1(gf_pre)
        gt4 = self.out1(gt4)
        edge1 = self.out2(edge1)
        rf1_map = self.out3(rf1_map)
        return gt4, rf1_map, edge1
# net是尝试融入文本