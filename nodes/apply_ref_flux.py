from ..flux.model import inject_flux
from ..flux.layers import inject_blocks


class ApplyRefFluxNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "fluxtapoz"

    def apply(self, model):
        # if hasattr(model.model.diffusion_model, 'is_ref') and model.model.diffusion_model.is_ref:
        #     return (model,)
        inject_flux(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)


class ConfigureRefFluxNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "latent": ("LATENT",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            "sigmas": ("SIGMAS",)
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "fluxtapoz"

    def apply(self, 
              model, 
              latent, 
              start_percent,
              end_percent,
              strength,
              sigmas):
        model = model.clone()
        sigma_to_percent = { sigma.item(): idx/len(sigmas) for idx, sigma in enumerate(sigmas)}
        sigma_to_step = { sigma.item(): idx for idx, sigma in enumerate(sigmas)}
        
        transformer_options = model.model_options.get('transformer_options', {})
        transformer_options = { **transformer_options }
        process_latent_in = model.get_model_object("process_latent_in")
        transformer_options['REF_OPTIONS'] = {
            'ref_latent': process_latent_in(latent['samples']),
            'start_percent': start_percent,
            'end_percent': end_percent,
            'sigma_to_percent': sigma_to_percent,
            'sigma_to_step': sigma_to_step,
            'strength': strength,
            'sigmas': sigmas,
        }
        model.model_options['transformer_options'] = transformer_options

        return (model,)
