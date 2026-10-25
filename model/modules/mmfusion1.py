import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from typing import List, Dict


# 移除所有 flash_attn 相关导入
# from flash_attn.bert_padding import unpad_input, pad_input
# try:
#     from .flash_attn import FlashCrossAttention
# except ImportError:
#     from flash_attn import FlashCrossAttention

class MultiModalFusion(nn.Module):
    def __init__(self, img_dim: int, text_dim: int) -> None:
        super(MultiModalFusion, self).__init__()

        # self.config = config

        self.img_norm = nn.LayerNorm(img_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        # self.cross_modal_attention = MultiModalAttention(
        #     query_dim=img_dim,
        #     key_dim=text_dim,
        #     value_dim=text_dim,
        #     head_dim=64,
        #     num_heads=8,
        #     dropout=0.1
        # )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=8,  # 8头注意力，需img_dim能被8整除（64/8=8）
            dropout=0.1,
            batch_first=True  # 输入格式 [B, seq_len, dim]
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(img_dim, img_dim * 3, bias=True)
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, img_x_lst, text_x: Dict[str, Tensor]) -> any:
        """Input
        Args:
            img_x_lst (List[Tensor]): list of image features, each shape like [B, C, H, W]
            text_x (Dict[str, Tensor]): text feature list.
        Returns:
            ...
        """
        new_img_x_lst = []
        # if img_x_lst isinstance List[Tensor]:
        # for img_x in img_x_lst:
        #     B, C, H, W = img_x.size()
        #     img_x = img_x.flatten(-2).permute(0, 2, 1)  # (B, C, H, W) -> (B, H*W, C)
        #     # mask = (~text_x['masks']).to(torch.bool)  # 改为bool类型（原生注意力要求）
        #     attn_w, _ = self.cross_attn(self.img_norm(img_x), self.text_norm(text_x['refs']),
        #                                         self.text_norm(text_x['refs']))
        #     # attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x['refs']),
        #     #                                     self.text_norm(text_x['refs']), attention_mask=mask), key_padding_mask=mask
        #     ref_embeds = text_x['ref_embeds']
        #     shift, scale, gate = self.adaLN_modulation(ref_embeds).chunk(3, dim=1)
        #     attn_w = gate.unsqueeze(1) * (attn_w * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        #     img_x = attn_w + img_x
        #     img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
        #     new_img_x_lst.append(img_x)
        
        img_x = img_x_lst
        B, C, H, W = img_x.size()
        img_x = img_x.flatten(-2).permute(0, 2, 1)  # (B, C, H, W) -> (B, H*W, C)
        # mask = (~text_x['masks']).to(torch.bool)  # 改为bool类型（原生注意力要求）
        attn_w, _ = self.cross_attn(self.img_norm(img_x), self.text_norm(text_x['refs']),self.text_norm(text_x['refs']))
        # attn_w = img_x
        # wo_T_s
        # attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x['refs']),
        #                                     self.text_norm(text_x['refs']), attention_mask=mask), key_padding_mask=mask
        # ref_embeds = text_x['ref_embeds']
        # shift, scale, gate = self.adaLN_modulation(ref_embeds).chunk(3, dim=1)
        # attn_w = gate.unsqueeze(1) * (attn_w * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        img_x = attn_w + img_x
        img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
        new_img_x_lst.append(img_x)

        return new_img_x_lst
class MultiModalAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, head_dim: int, num_heads: int, value_dim: int = None,
                 dropout: float = 0.1) -> None:
        super(MultiModalAttention, self).__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.embed_dim = head_dim * num_heads

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim

        self.query_proj = nn.Linear(self.query_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.key_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.value_dim, self.embed_dim)

        # 替换 FlashCrossAttention 为 PyTorch 原生 MultiheadAttention
        self.mh_crossattn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 关键：指定batch在第一维（和原逻辑对齐）
        )

        self.out_proj = nn.Linear(self.embed_dim, query_dim)

        self._init_parameters()

    def _init_parameters(self):
        xavier_uniform_(self.query_proj.weight);
        self.query_proj.bias.data.zero_()
        xavier_uniform_(self.key_proj.weight);
        self.key_proj.bias.data.zero_()
        xavier_uniform_(self.value_proj.weight);
        self.value_proj.bias.data.zero_()
        xavier_uniform_(self.out_proj.weight);
        self.out_proj.bias.data.zero_()

    def forward(self, q, k, v=None, attention_mask=None):
        B, QLEN, QDIM = q.shape
        B, KLEN, KDIM = k.shape

        q_proj = self.query_proj(q)  # (B, QLEN, QDIM) -> (B, QLEN, embed_dim)
        k_proj = self.key_proj(k)  # (B, KLEN, KDIM) -> (B, KLEN, embed_dim)
        v_proj = self.value_proj(v if v is not None else k)

        # 完全删除 FlashAttention 专属的 unpad/pad/varlen 逻辑
        # 直接调用原生 MultiheadAttention
        out, _ = self.mh_crossattn(
            query=q_proj,
            key=k_proj,
            value=v_proj,
            key_padding_mask=attention_mask  # 原生注意力的mask是bool类型（True表示padding）
        )

        out = self.out_proj(out)
        return out