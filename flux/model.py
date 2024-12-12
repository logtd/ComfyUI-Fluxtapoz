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
        ref_config: Dict[Any, Any] | None = None,
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

        ref_pes = None
        if ref_config is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
            # ids = torch.cat((txt_ids[:-1], img_ids), dim=1)
            # pe = self.pe_embedder(ids)

            ref_pes = []
            for ref_img_id in ref_config['img_ids']:
                ref_ids = torch.cat([txt_ids[:1], ref_img_id], dim=1)
                ref_pe = self.pe_embedder(ref_ids)
                ref_pes.append(ref_pe)
            ref_config['ref_pes'] = ref_pes
            del ref_config['img_ids']
        else:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, ref_config=ref_config, timestep=timesteps, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add
                        if ref_pes is not None:
                            img[1:] += add
                        #     img[-1:, txt.shape[1] :txt.shape[1]+4096, ...] += add
                        #     img[:, 4096*1:4096*2] += add
                            # img[:, 4096*2:4096*3] += add

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe, ref_config=None, timestep=timesteps, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:1, txt.shape[1] :, ...] += add
                        # img[:1, txt.shape[1] :txt.shape[1]+4096, ...] += add
                        if ref_pes is not None:
                            img[-1:, txt.shape[1] :, ...] += add
                        #     img[:, txt.shape[1]+4096*1 :txt.shape[1]+4096*2, ...] += add
                            # img[:, txt.shape[1]+4096*2 :txt.shape[1]+4096*3, ...] += add
                        
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

        ref_options = transformer_options.get('REF_OPTIONS', None)

        perform_ref = False
        ref_config = None
        step = None
        if ref_options is not None:
            ref_start_percent = ref_options.get('start_percent', 0)
            ref_end_percent = ref_options.get('end_percent', -1)
            sigma_percents = ref_options.get('sigma_to_percent', {})
            step_percent = sigma_percents[timestep[0].item()]
            perform_ref = ref_start_percent <= step_percent < ref_end_percent
            sigma_to_step = ref_options.get('sigma_to_step', {})
            step = sigma_to_step[timestep[0].item()]

        ref_img_ids = None
        if perform_ref:
            # ref
            ref_config = {** ref_options }
            ref_config['step'] = step
            ref_latent = ref_options['ref_latent']
            ref_latent = ref_latent.to(x.device)
            sigma = ref_options.get('sigmas', [])[sigma_to_step[timestep[0].item()]].to(x.device)
            use_sigma = ref_options['use_sigmas'][step].to(x.device)

            generator = torch.Generator()
            generator.manual_seed(0)
            noise = torch.randn(ref_latent.shape, generator=generator).to(x.device)

            ref_latent =  add_noise_flux(ref_latent, noise, use_sigma)
            # noise = torch.randn_like(ref_latent)
            # ref_latent = add_noise(ref_latent, noise, sigma)
            ref_latent = rearrange(ref_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # img = torch.cat([img, ref_latent], dim=0) # this won't work with cfg
            # horizontal translation
            diff = 2
            ref_img_id1 = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, diff+w_len, diff+2*w_len)
            ref_img_id2 = self._get_img_ids(x, bs, h_len, w_len, h_len+diff, diff+2*h_len, 0, w_len)
            # ref_img_id2 = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, 0, 2*w_len)
            ref_img_ids = [ref_img_id1, ref_img_id2]
            ref_config['img_ids'] = ref_img_ids

            # img_ids_orig = self._get_img_ids(x, bs, h_len*2, w_len, 0, h_len*2, 0, w_len)
            img = torch.cat([img, ref_latent], dim=0)

            timestep = torch.concat([timestep, use_sigma.to(timestep.device).expand(1)])

            # timestep = timestep.repeat(2)
            txt_ids = txt_ids.repeat(2, 1, 1)
            context = context.repeat(2, 1, 1)
            y = y.repeat(2, 1)
            guidance = guidance.repeat(2)
            # guidance = torch.concat([guidance, torch.Tensor([0]).to(x.device)])
            img_ids_orig = img_ids_orig.repeat(2,1,1)

        rave_config = transformer_options.get('RAVE', None)
        if rave_config is not None:
            grid_size = rave_config['grid_size']
            img_ids_orig = self._get_img_ids(x, bs, h_len*grid_size, w_len*grid_size, 0, h_len*grid_size, 0, w_len*grid_size)

        transformer_options['txt_size'] = context.shape[1]

        out = self.forward_orig(img, img_ids_orig, context, txt_ids, timestep, y, guidance, control, transformer_options=transformer_options, ref_config=ref_config)

        if perform_ref:
            out = out[:-1]
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]


def inject_flux(diffusion_model: OriginalFlux):
    diffusion_model.__class__ = Flux
    # diffusion_model.is_ref = True
    return diffusion_model