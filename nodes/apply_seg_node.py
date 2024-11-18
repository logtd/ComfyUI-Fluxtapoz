
import math

from einops import rearrange
import torch
import torch.nn.functional as F

import comfy.model_patcher
import comfy.samplers


DEFAULT_SEG_FLUX = { 'double': set([]), 'single': set(['0'])}


def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class SEGAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "blur": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 999.0, "step": 0.01, "round": 0.01}),
                "inf_blur": ("BOOLEAN", {"default": False} ),
            },
            "optional": {
                "attn_override": ("ATTN_OVERRIDE",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "fluxtapoz/attn"

    def patch(self, model, scale, blur, inf_blur, attn_override=DEFAULT_SEG_FLUX):
        m = model.clone()

        def seg_attention(q, extra_options, txt_shape=256):
            _, _, sequence_length, _ = q.shape
            shape = extra_options['original_shape']
            patch_size = extra_options['patch_size']
            oh, ow = shape[-2:]
            h = oh // patch_size
            q_img = q[:, :, txt_shape:]
            num_heads = q_img.shape[1]
            q_img = rearrange(q_img, 'b a (h w) d -> b (a d) w h', h=h)
            if not inf_blur:
                kernel_size = math.ceil(6 * blur) + 1 - math.ceil(6 * blur) % 2
                q_img = gaussian_blur_2d(q_img, kernel_size, blur)
            else:
                q_img = q_img.mean(dim=(-2, -1), keepdim=True)
            q_img = rearrange(q_img, 'b (a d) w h -> b a (h w) d', a=num_heads)
            q[:,:, txt_shape:] = q_img
            return q

        def post_cfg_function(args):
            model = args["model"]

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            if scale == 0 or blur == 0:
                return uncond_pred + (cond_pred - uncond_pred)
            
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]
            # Hack since comfy doesn't pass in conditionals and unconditionals to cfg_function
            # and doesn't pass in cond_scale to post_cfg_function
            len_conds = 1 if args.get('uncond', None) is None else 2 
            
            for block_idx in attn_override['double']:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, seg_attention, f"double", "post_q", int(block_idx))

            for block_idx in attn_override['single']:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, seg_attention, f"single", "post_q", int(block_idx))

            (seg,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            if len_conds == 1:
                return cond_pred + scale * (cond_pred - seg)

            return cond_pred + (scale-1.0) * (cond_pred - uncond_pred) + scale * (cond_pred - seg)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
