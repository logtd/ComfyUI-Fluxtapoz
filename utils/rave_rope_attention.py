import random
import torch
from torch import Tensor
from einops import rearrange

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention

from .rave_utils import grid_to_list, list_to_grid, shuffle_indices, shuffle_tensors2


def padding_count(n_frames, grid_frame_count):
    remainder = n_frames % grid_frame_count
    if remainder == 0:
        return 0
    else:
        difference = grid_frame_count - remainder
        return difference


def attention(q: torch.Tensor, k: Tensor, v: Tensor, pe: Tensor,) -> Tensor:
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def rave_rope_attention(q, k, v, txt_q, txt_k, txt_v, pe, extra_options, n_heads, txt_shape):
    q = rearrange(q, 'b n r d -> b r (n d)')
    k = rearrange(k, 'b n r d -> b r (n d)')
    v = rearrange(v, 'b n r d -> b r (n d)')
    rave_opts = extra_options.get('RAVE', {})
    grid_size = rave_opts.get('grid_size', 2)
    seed = rave_opts.get('seed', 1)
    batch_size, sequence_length, dim = q.shape
    len_conds = len(extra_options['cond_or_uncond'])
    n_frames = batch_size // len_conds
    original_n_frames = n_frames

    grid_frame_count = grid_size * grid_size
    n_padding_frames = padding_count(n_frames, grid_frame_count)
    if n_padding_frames > 0:
        random.seed(seed)
        cond_qs = []
        cond_ks = []
        cond_vs = []
        padding_frames = [random.randint(
            0, n_frames-1) for _ in range(n_padding_frames)]
        for cond_idx in range(len_conds):
            start, end = cond_idx*n_frames, (cond_idx+1)*n_frames
            cond_q = q[start:end]
            cond_q = torch.cat([cond_q, cond_q[padding_frames]])
            cond_qs.append(cond_q)
            cond_k = k[start:end]
            cond_k = torch.cat([cond_k, cond_k[padding_frames]])
            cond_ks.append(cond_k)
            cond_v = v[start:end]
            cond_v = torch.cat([cond_v, cond_v[padding_frames]])
            cond_vs.append(cond_v)

        q = torch.cat(cond_qs)
        k = torch.cat(cond_ks)
        v = torch.cat(cond_vs)

    n_frames = n_frames + n_padding_frames

    # get h,w
    shape = extra_options['original_shape']
    oh, ow = shape[-2:]
    ratio = oh/ow
    d = sequence_length
    w = int((d/ratio)**(0.5))
    h = int(d/w)

    q = rearrange(q, 'b (h w) c -> b h w c', h=h, w=w)
    k = rearrange(k, 'b (h w) c -> b h w c', h=h, w=w)
    v = rearrange(v, 'b (h w) c -> b h w c', h=h, w=w)

    target_indexes = shuffle_indices(n_frames, seed=seed)

    original_indexes = list(range(n_frames))
    qs = []
    ks = []
    vs = []

    for i in range(len_conds):
        start, end = i*n_frames, (i+1)*n_frames
        q[start:end] = shuffle_tensors2(
            q[start:end], original_indexes, target_indexes)
        qs.append(list_to_grid(q[start:end], grid_size))
        k[start:end] = shuffle_tensors2(
            k[start:end], original_indexes, target_indexes)
        ks.append(list_to_grid(k[start:end], grid_size))
        v[start:end] = shuffle_tensors2(
            v[start:end], original_indexes, target_indexes)
        vs.append(list_to_grid(v[start:end], grid_size))

    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)

    q = rearrange(q, 'b h w (n d) -> b n (h w) d', n=n_heads)
    k = rearrange(k, 'b h w (n d) -> b n (h w) d', n=n_heads)
    v = rearrange(v, 'b h w (n d) -> b n (h w) d', n=n_heads)
    # out = optimized_attention(q, k, v, n_heads, None)
    txt_q = txt_q[:len(q)]
    txt_k = txt_k[:len(q)]
    txt_v = txt_v[:len(q)]
    q = torch.cat((txt_q, q), dim=2)
    k = torch.cat((txt_k, k), dim=2)
    v = torch.cat((txt_v, v), dim=2)
    out = attention(q, k, v, pe)

    txt_attn, out = out[:, :txt_shape], out[:, txt_shape:]
    txt_attn = txt_attn.repeat(grid_size*grid_size, 1, 1)

    gh, gw = grid_size*h, grid_size*w
    out = rearrange(out, 'b (h w) c -> b h w c', h=gh, w=gw)
    out = grid_to_list(out, grid_size)
    out = rearrange(out, 'b h w c -> b (h w) c')

    outs = []
    for i in range(len_conds):
        start, end = i*n_frames, (i+1)*n_frames
        cond_out = shuffle_tensors2(
            out[start:end], target_indexes, original_indexes)
        cond_out = cond_out[:original_n_frames]
        outs.append(cond_out)

    return txt_attn, torch.cat(outs)
