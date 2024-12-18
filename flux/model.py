#Original code can be found on: https://github.com/black-forest-labs/flux
from typing import Any, Dict, List
import torch
from torch import Tensor, nn

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.model import Flux as OriginalFlux

from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..utils.noise_utils import add_noise_flux, add_noise


class Flux(OriginalFlux):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options = {},
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add
                        
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
    def _get_img_ids(self, x, bs, h_len, w_len, h_start, h_end, w_start, w_end):
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(h_start, h_end - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(w_start, w_end - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        return img_ids

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        transformer_options['original_shape'] = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
        transformer_options['patch_size'] = patch_size

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        regional_conditioning = transformer_options.get('patches', {}).get('regional_conditioning', None)
        if regional_conditioning is not None:
            region_cond = regional_conditioning[0](transformer_options)
            if region_cond is not None:
                context = torch.cat([context, region_cond.to(context.dtype)], dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        img_ids_orig = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, 0, w_len)

        transformer_options['txt_size'] = context.shape[1]

        out = self.forward_orig(img, img_ids_orig, context, txt_ids, timestep, y, guidance, control, transformer_options=transformer_options)

        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]


def inject_flux(diffusion_model: OriginalFlux):
    diffusion_model.__class__ = Flux
    # diffusion_model.is_ref = True
    return diffusion_model