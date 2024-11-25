import torch


class RegionalStyleModelApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1, "min": -1000.0, "max": 1000.0, "step": 0.01, "round": 0.01}),
                             },
                             "optional": {
                                 "conditioning": ("CONDITIONING", ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "fluxtapoz"

    def apply_stylemodel(self, clip_vision_output, style_model, strength, conditioning=None):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0) * strength
        c = []
        if conditioning is not None:
            for t in conditioning:
                n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
                c.append(n)
        else:
            c.append([cond, None])
        return (c, )
