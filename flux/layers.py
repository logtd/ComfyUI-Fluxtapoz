from typing import List
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import SingleStreamBlock as OriginalSingleStreamBlock, DoubleStreamBlock as OriginalDoubleStreamBlock


import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.modules.attention import optimized_attention

# from ..utils.rave_rope_attention import rave_rope_attention
# from ..utils.rave_attention import rave_attention
# from ..utils.joint_attention import joint_attention


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, skip_rope: bool= False) -> Tensor:
    if not skip_rope:
        q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def ref_attention2(q: Tensor, k:Tensor, v:Tensor, pe: Tensor, ref_pes: List[Tensor], transformer_options, timesteps) -> Tensor:
    # rope tgt
    timesteps = timesteps.item()
    q_tgt, k_tgt = apply_rope(q[[0]], k[[0]], pe)
    strength = transformer_options['REF_OPTIONS']['strength']

    # rope refs
    q_ref = q[[-1],:,256:,:]
    k_ref = k[[-1],:,256:,:]
    
    _, k_ref1 = apply_rope(q_ref, k_ref, ref_pes[0])
    # k_ref1 = k_ref1[[-1], :, 256:, :] * strength

    # if 0.85 <= timesteps:

    #     k_ref1[:, :, :, 16+0 :16+0 +28+8-1] *= 0.1
    #     k_ref1[:, :, :, 16+56:16+56+28+8-1] *= 0.1

    # elif 0.6 <= timesteps < 0.85:
    
    #     k_ref1[:, :, :, 16+0 :16+0 +28-20-1] *= 0.1
    #     k_ref1[:, :, :, 16+56:16+56+28-20-1] *= 0.1

    # _, k_ref2 = apply_rope(q_ref, k_ref, ref_pes[1])
    # k_ref2 = k_ref2[[-1], :, 256:, :]
    v_ref = v[[-1],:, 256:, :]

    k_tgt = torch.cat([k_tgt, k_ref1], dim=2)
    v_tgt = torch.cat([v[[0]], v_ref], dim=2)
    tgt_out = attention(q_tgt, k_tgt, v_tgt, pe, skip_rope=True)

    ref_out = attention(q[[-1]], k[[-1]], v[[-1]], pe)

    return torch.cat([tgt_out, ref_out], dim=0)


def ref_attention(q: Tensor, k:Tensor, v:Tensor, pe: Tensor, ref_pes: List[Tensor], transformer_options, timesteps) -> Tensor:
    # rope tgt
    timesteps = timesteps.item()
    strength = transformer_options['REF_OPTIONS']['strength']

    # rope refs
    q_ref1 = torch.cat([q[[0]], q[[-1], :,256:,:]], dim=2)
    k_ref1 = torch.cat([k[[0]], k[[-1], :,256:,:]*strength], dim=2)
    v_ref = torch.cat([v[[0]], v[[-1], :,256:,:]], dim=2)
    
    _, k_ref1 = apply_rope(q_ref1, k_ref1, ref_pes[1])

    tgt_out = attention(q[[0]], k_ref1, v_ref, pe, skip_rope=True)

    ref_out = attention(q[[-1]], k[[-1]], v[[-1]], pe)

    return torch.cat([tgt_out, ref_out], dim=0)


class DoubleStreamBlock(OriginalDoubleStreamBlock):
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, ref_pes: None | List[Tensor], timestep, transformer_options={}):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)


        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        ref_options = transformer_options.get('REF_OPTIONS', None)
        if ref_options is not None and ref_pes is not None:
            attn = ref_attention(q, k, v, pe, ref_pes, transformer_options, timestep[0])
        else:
            attn = attention(q, k, v, pe=pe)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        txt_attn = txt_attn[0:1].repeat(img_attn.shape[0], 1, 1)

        # img_attn.shape [16, 2304, 3072]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(OriginalSingleStreamBlock):
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, ref_pes, timestep, transformer_options={}) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        ref_options = transformer_options.get('REF_OPTIONS', None)
        if ref_options is not None and ref_pes is not None:
            attn = ref_attention(q, k, v, pe, ref_pes, transformer_options, timestep[0])
        else:
            attn = attention(q, k, v, pe=pe)
        # txt_attn, img_attn = attn[:, :256], attn[:, 256:]
        
        # txt_attn = temporal_attention(txt_attn, self.num_heads, transformer_options)
        # attn[:, :256] = txt_attn

        # img_attn = interframe_attention(img_attn, self.num_heads, transformer_options)
        # attn[:, 256:] = img_attn

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


def inject_blocks(diffusion_model):
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = DoubleStreamBlock
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = SingleStreamBlock
        block.idx = i

    return diffusion_model