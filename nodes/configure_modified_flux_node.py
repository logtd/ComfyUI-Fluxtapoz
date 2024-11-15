from ..flux.model import inject_flux
from ..flux.layers import inject_blocks


class ConfigureModifiedFluxNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "fluxtapoz"
    FUNCTION = "apply"

    def apply(self, model):
        # if hasattr(model.model.diffusion_model, 'is_ref') and model.model.diffusion_model.is_ref:
        #     return (model,)
        inject_flux(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)

