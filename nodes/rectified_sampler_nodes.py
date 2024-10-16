import torch
from tqdm import trange

from comfy.samplers import KSAMPLER


def get_sample_forward(gamma):
    # Controlled Forward ODE (Algorithm 1)
    @torch.no_grad()
    def sample_forward(model, y0, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        Y = y0.clone()
        y1 = torch.randn_like(Y)
        N = len(sigmas)
        s_in = y0.new_ones([y0.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            t_i = i / N  

            # 6. Unconditional Vector field uti(Yti) = u(Yti, ti, Φ(“”); φ)
            unconditional_vector_field = model(Y, s_in * sigmas[i], **extra_args) # this implementation takes sigma instead of timestep
            
            # 7.Conditional Vector field  uti(Yti|y1) = (y1−Yti)/1−ti
            conditional_vector_field = (y1-Y)/(1-t_i)
            
            # 8. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
            controlled_vector_field = unconditional_vector_field + gamma * (conditional_vector_field - unconditional_vector_field)
            
            # 9. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
            Y = Y + controlled_vector_field * (sigmas[i+1] - sigmas[i])

            if callback is not None:
                callback({'x': Y, 'denoised': Y, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return Y

    return sample_forward


def get_sample_reverse(latent_image, eta, start_time, end_time):
    # Controlled Reverse ODE (Algorithm 2)
    @torch.no_grad()
    def sample_reverse(model, y1, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        X = y1.clone()
        N = len(sigmas)
        y0 = latent_image.clone().to(y1.device)
        s_in = y0.new_ones([y0.shape[0]])
        for i in trange(N - 1, disable=disable):
            t_i = i / N  

            # 5. Unconditional Vector field uti(Xti) = -u(Xti, 1-ti, Φ(“prompt”); φ)
            unconditional_vector_field = -model(X, sigmas[i]*s_in, **extra_args) # this implementation takes sigma instead of timestep
            
            # 6.Conditional Vector field  uti(Xti|y0) = (y0−Xti)/(1−ti)
            conditional_vector_field = (y0-X)/(1-t_i)
            
            # 7. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
            controlled_vector_field = unconditional_vector_field
            if start_time <= i < end_time:
                controlled_vector_field = unconditional_vector_field + eta * (conditional_vector_field - unconditional_vector_field)
            
            # 8. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
            X = X + controlled_vector_field * (sigmas[i] - sigmas[i+1])

            if callback is not None:
                callback({'x': X, 'denoised': X, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return X
    
    return sample_reverse


class FluxForwardODESamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "gamma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "flux"

    def build(self, gamma):
        sampler = KSAMPLER(get_sample_forward(gamma))

        return (sampler, )


class FluxReverseODESamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "latent_image": ("LATENT",),
            "eta": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 100.0, "step": 0.01}),
            "start_step": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            "end_step": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "flux"

    def build(self, model, latent_image, eta, start_step, end_step):
        process_latent_in = model.get_model_object("process_latent_in")
        latent_image = process_latent_in(latent_image['samples'])
        sampler = KSAMPLER(get_sample_reverse(latent_image, eta, start_step, end_step))

        return (sampler, )
