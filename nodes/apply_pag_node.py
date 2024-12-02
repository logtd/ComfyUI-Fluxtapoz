import math

from einops import rearrange
import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_patcher
import comfy.samplers


DEFAULT_PAG_FLUX = { 'double': set([]), 'single': set(['0'])}


class PAGAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
            },
            "optional": {
                "attn_override": ("ATTN_OVERRIDE",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "fluxtapoz/attn"

    def patch(self, model, scale, attn_override=DEFAULT_PAG_FLUX):
        m = model.clone()

        def pag_mask(q, extra_options, mask=None):
            # From diffusers implementation
            seq_len = q.size(2)
            shape = extra_options['original_shape']
            patch_size = extra_options['patch_size']
            oh, ow = shape[-2:]
            h = oh // patch_size
            
            # Get text token length from model config
            text_tokens = extra_options.get('text_tokens', seq_len - (h * h))
            
            # Create a full mask with all entries set to 0
            full_mask = torch.zeros((seq_len, seq_len), device=q.device, dtype=q.dtype)

            # Set the attention value between image patches to -inf
            identity_block_size = seq_len - text_tokens
            full_mask[:identity_block_size, :identity_block_size] = float("-inf")

            # Set the diagonal of the attention value between image patches to 0
            full_mask[:identity_block_size, :identity_block_size].fill_diagonal_(0)

            # Expand the mask to match the attention weights shape
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

            return full_mask

        def post_cfg_function(args):
            model = args["model"]

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            len_conds = 1 if args.get('uncond', None) is None else 2 
            if scale == 0:
                if len_conds == 1:
                    return cond_pred
                return uncond_pred + (cond_pred - uncond_pred)
            
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]
            # Hack since comfy doesn't pass in conditionals and unconditionals to cfg_function
            # and doesn't pass in cond_scale to post_cfg_function
            
            for block_idx in attn_override['double']:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, pag_mask, f"double", "mask_fn", int(block_idx))

            for block_idx in attn_override['single']:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, pag_mask, f"single", "mask_fn", int(block_idx))

            (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            if len_conds == 1:
                return cond_pred + scale * (cond_pred - pag)

            return cond_pred + (scale-1.0) * (cond_pred - uncond_pred) + scale * (cond_pred - pag)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
