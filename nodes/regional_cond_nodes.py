import torch

import comfy.sd
import comfy.model_sampling


DEFAULT_REGIONAL_ATTN = {
    'double': [i for i in range(1, 19, 2)],
    'single': [i for i in range(1, 38, 2)]
}


class RegionalMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, q, transformer_options, *args, **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.mask
        
        return None
    

class RegionalConditioning(torch.nn.Module):
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, transformer_options, *args,  **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.region_cond
        return None


class CreateRegionalCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "cond": ("CONDITIONING",),
            "mask": ("MASK",),
        }, "optional": {
            "prev_regions": ("REGION_COND",),
        }}

    RETURN_TYPES = ("REGION_COND",)
    FUNCTION = "create"

    CATEGORY = "fluxtapoz"

    def create(self, cond, mask, prev_regions=[]):
        prev_regions = [*prev_regions]
        prev_regions.append({
            'mask': mask,
            'cond': cond[0][0]
        })

        return (prev_regions,)


class ApplyRegionalCondsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "region_conds": ("REGION_COND",),
            "latent": ("LATENT",),
            "start_percent": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",)
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "fluxtapoz"

    def patch(self, model, region_conds, latent, start_percent, end_percent, attn_override=DEFAULT_REGIONAL_ATTN):
        model = model.clone()

        latent = latent['samples']
        b, c, h, w = latent.shape
        h //=2
        w //=2

        img_len = h*w

        regional_conditioning = torch.cat([region_cond['cond'] for region_cond in region_conds], dim=1)
        text_len = 256 + regional_conditioning.shape[1]

        regional_mask = torch.zeros((text_len + img_len, text_len + img_len), dtype=torch.bool)

        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool)
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool)

        region_conds = [
            { 
                'mask': torch.ones((1, h, w), dtype=torch.float16),
                'cond': torch.ones((1, 256, 4096), dtype=torch.float16)
            },
            *region_conds
        ]

        current_seq_len = 0
        for region_cond_dict in region_conds:
            region_cond = region_cond_dict['cond']
            region_mask = 1 - region_cond_dict['mask'][0]
            region_mask = torch.nn.functional.interpolate(region_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, region_cond.size(1))
            next_seq_len = current_seq_len + region_cond.shape[1]

            # txt attends to itself
            regional_mask[current_seq_len:next_seq_len, current_seq_len:next_seq_len] = True

            # txt attends to corresponding regional img
            regional_mask[current_seq_len:next_seq_len, text_len:] = region_mask.transpose(-1, -2)

            # regional img attends to corresponding txt
            regional_mask[text_len:, current_seq_len:next_seq_len] = region_mask

            # regional img attends to corresponding regional img
            img_size_masks = region_mask[:, :1].repeat(1, img_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))
            
            current_seq_len = next_seq_len

        background_masks = torch.logical_not(union_masks)
        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)
        regional_mask[text_len:, text_len:] = background_and_self_attend_masks

        # Patch
        regional_mask = RegionalMask(regional_mask, start_percent, end_percent)
        regional_conditioning = RegionalConditioning(regional_conditioning, start_percent, end_percent)

        model.set_model_patch(regional_conditioning, 'regional_conditioning')

        for block_idx in attn_override['double']:
            model.set_model_patch_replace(regional_mask, f"double", "mask_fn", int(block_idx))

        for block_idx in attn_override['single']:
            model.set_model_patch_replace(regional_mask, f"single", "mask_fn", int(block_idx))

        return (model,)
