import torch
from tqdm import trange

from comfy.samplers import KSAMPLER


@torch.no_grad()
def sample_inverse(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + denoised * dt
    return x


class FluxInverseSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "fluxtapoz"

    def build(self):

        sampler = KSAMPLER(sample_inverse)

        return (sampler, torch.Tensor([0]))
