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
        attn_w, _ = self.cross_attn(self.img_norm(img_x), self.text_norm(text_x['refs']),
                                            self.text_norm(text_x['refs']))
        # attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x['refs']),
        #                                     self.text_norm(text_x['refs']), attention_mask=mask), key_padding_mask=mask
        ref_embeds = text_x['ref_embeds']
        shift, scale, gate = self.adaLN_modulation(ref_embeds).chunk(3, dim=1)
        attn_w = gate.unsqueeze(1) * (attn_w * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        img_x = attn_w + img_x
        img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
        new_img_x_lst.append(img_x)
        # for img_x in img_x_lst:
        #     B, C, H, W = img_x.size()
        #     img_x = img_x.flatten(-2).permute(0, 2, 1)  # (B, C, H, W) -> (B, H*W, C)
        #     attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x),
        #                                         self.text_norm(text_x))
        #     img_x = attn_w + img_x
        #     img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
        #     new_img_x_lst.append(img_x)

        return new_img_x_lst
# class MultiModalFusion(nn.Module):
#     def __init__(self, img_dim: int, text_dim: int) -> None:
#         super(MultiModalFusion, self).__init__()

#         self.img_norm = nn.LayerNorm(img_dim)
#         self.text_norm = nn.LayerNorm(text_dim)

#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=img_dim,
#             num_heads=8,  # 8头注意力，需img_dim能被8整除（64/8=8）
#             dropout=0.1,
#             batch_first=True  # 输入格式 [B, seq_len, dim]
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(img_dim, img_dim * 3, bias=True)
#         )

#         self._init_parameters()

#     def _init_parameters(self):
#         nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

#     def forward(self, img_x_lst, text_x: Dict[str, Tensor]) -> any:
#         """Input
#         Args:
#             img_x_lst (List[Tensor]): list of image features, each shape like [B, C, H, W]
#             text_x (Dict[str, Tensor]): text feature list.
#         Returns:
#             ...
#         """
#         new_img_x_lst = []
#         img_x = img_x_lst
#         B, C, H, W = img_x.size()
#         img_x = img_x.flatten(-2).permute(0, 2, 1)  # (B, C, H, W) -> (B, H*W, C)
#         # mask = (~text_x['masks']).to(torch.bool)  # 改为bool类型（原生注意力要求）
#         attn_w, _ = self.cross_attn(self.img_norm(img_x), self.text_norm(text_x['refs']),
#                                             self.text_norm(text_x['refs']))
#         # attn_w = self.cross_modal_attention(self.img_norm(img_x), self.text_norm(text_x['refs']),
#         #                                     self.text_norm(text_x['refs']), attention_mask=mask), key_padding_mask=mask
#         ref_embeds = text_x['ref_embeds']
#         shift, scale, gate = self.adaLN_modulation(ref_embeds).chunk(3, dim=1)
#         attn_w = gate.unsqueeze(1) * (attn_w * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
#         img_x = attn_w + img_x
#         img_x = img_x.permute(0, 2, 1).reshape(B, C, H, W)
#         new_img_x_lst.append(img_x)

#         return new_img_x_lst
# class MultiModalFusion(nn.Module):
#     def __init__(self, img_dim: int = 64, text_dim: int = 64, config=None) -> None:
#         super(MultiModalFusion, self).__init__()
#         self.img_dim = img_dim
#         self.text_dim = text_dim
        
#         # 1. 文本特征预处理：768→64（匹配视觉维度）
#         self.text_proj = nn.Sequential(
#             nn.Linear(768, text_dim),
#             nn.LayerNorm(text_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#         # 2. 视觉特征归一化
#         self.img_norm = nn.LayerNorm(img_dim)
        
#         # 3. 跨模态交叉注意力（核心融合层）
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=img_dim,
#             num_heads=8,  # 8头注意力，需img_dim能被8整除（64/8=8）
#             dropout=0.1,
#             batch_first=True  # 输入格式 [B, seq_len, dim]
#         )
        
#         # 4. 自适应调制（增强融合效果）
#         self.adaLN = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(text_dim, img_dim * 3)  # 输出shift/scale/gate
#         )
        
#         # 初始化参数
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0)

#     def _prepare_text_features(self, text_feature: Tensor) -> Dict[str, Tensor]:
#         """
#         预处理文本特征：
#         输入：text_feature [B, 64, 768] → 输出字典包含：
#         - refs: 投影后的文本特征 [B, 64, 64]
#         - masks: 文本padding mask [B, 64]（假设无padding，全1）
#         - ref_embeds: 文本全局特征 [B, 64]（mean池化）
#         """
#         B, seq_len, _ = text_feature.shape
        
#         # 降维：768→64
#         text_proj = self.text_proj(text_feature)  # [B, 64, 64]
        
#         # 生成mask（假设文本无padding，全为有效token）
#         masks = torch.ones(B, seq_len, device=text_feature.device)  # [B, 64]
        
#         # 文本全局特征（mean池化）
#         ref_embeds = torch.mean(text_proj, dim=1)  # [B, 64]
        
#         return {
#             "refs": text_proj,          # [B, 64, 64] 文本序列特征
#             "masks": masks,            # [B, 64] 有效token掩码
#             "ref_embeds": ref_embeds   # [B, 64] 文本全局特征
#         }

#     def forward(self, img_x_lst: List[Tensor], text_feature: Tensor) -> List[Tensor]:
#         """
#         主前向逻辑：
#         Args:
#             img_x_lst: 视觉特征列表，每个元素 [B, 64, H, W]（如x1_proj/x2_proj/x3_proj/x4_proj）
#             text_feature: 文本特征 [B, 64, 768]
#         Returns:
#             融合后的视觉特征列表，维度与输入一致
#         """
#         # 1. 预处理文本特征
#         text_dict = self._prepare_text_features(text_feature)
#         refs = text_dict["refs"]          # [B, 64, 64]
#         masks = text_dict["masks"]        # [B, 64]
#         ref_embeds = text_dict["ref_embeds"]  # [B, 64]
        
#         # 2. 对每个尺度的视觉特征做融合
#         fused_img_lst = []
#         for img_x in img_x_lst:
#             B, C, H, W = img_x.shape  # [B, 64, H, W]
            
#             # 视觉特征重塑：[B, 64, H, W] → [B, H*W, 64]（适配注意力输入）
#             img_flat = img_x.flatten(2).permute(0, 2, 1)  # [B, H*W, 64]
#             img_flat_norm = self.img_norm(img_flat)       # 归一化
            
#             # 3. 跨模态交叉注意力（视觉query，文本key/value）
#             # mask: True表示padding（这里无padding，所以取反后全False）
#             attn_mask = (~masks.bool()).to(img_x.device)  # [B, 64]
#             attn_out, _ = self.cross_attn(
#                 query=img_flat_norm,
#                 key=refs,
#                 value=refs,
#                 key_padding_mask=attn_mask
#             )  # attn_out: [B, H*W, 64]
            
#             # 4. 自适应调制（用文本全局特征增强融合结果）
#             shift, scale, gate = self.adaLN(ref_embeds).chunk(3, dim=1)  # 各[B, 64]
#             attn_out = gate.unsqueeze(1) * (attn_out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
            
#             # 5. 残差连接 + 重塑回原维度
#             img_fused_flat = img_flat + attn_out  # 残差
#             img_fused = img_fused_flat.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 64, H, W]
            
#             fused_img_lst.append(img_fused)
        
#         return fused_img_lst
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

# 移除 FlashAttention 专属的 AttnTensorTools 类（已无用）
# class AttnTensorTools:
#     @classmethod
#     def to_multi_head_shape(cls, tensor: Tensor, bs: int, seq_len: int, num_heads: int, head_dim: int) -> Tensor:
#         return tensor.view(bs, seq_len, num_heads, head_dim).contiguous()


# 测试代码也同步修改（可选保留）
# if __name__ == "__main__":
#     # Test MultiModalAttention
#     ref_text = "Just a sentence for test!"
#     from transformers import CLIPTextModel, CLIPTokenizerFast
#     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#     tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
#     for p in text_encoder.parameters():
#         p.requires_grad_(False)
#     tokenized = tokenizer(ref_text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
#     encoded_text = text_encoder(**tokenized)
#     text_attention_mask = tokenized['attention_mask'].ne(1).bool()

#     text_feat_dict = {
#         'text_features': encoded_text.last_hidden_state,
#         'text_attention_mask': text_attention_mask,
#     }

#     from textproc import TextProcesser
#     text_proc = TextProcesser(512, 512)
#     text_feat_dict = text_proc(text_feat_dict)

#     mm_attention = MultiModalAttention(512, 512, 64, 8, 512).cuda()
#     img_x = torch.randn(1, 512, 22, 22)
#     img_x = img_x.flatten(-2).permute(0, 2, 1)
#     print("Image reshape: ", img_x.shape)
#     mask = (~text_feat_dict['masks']).to(torch.bool)
#     result = mm_attention(img_x.cuda(), text_feat_dict['refs'].cuda(), text_feat_dict['refs'].cuda(), mask.cuda())
#     print(result.cpu().shape)