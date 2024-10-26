
default_attn = {
    'inputs': [True] * 10,
    'input_idxs': list(range(10)),
    'middle_0': True,
    'outputs': [True] * 12,
    'output_idxs': list(range(12))
}


class ApplyFluxRaveAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "grid_size": ("INT", {"default": 3, "min": 1, "max": 10}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "optional": {
                    "attn_override": ("ATTN_OVERRIDE",)
                }
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "fluxtapoz"

    def apply(self, model, grid_size, seed, attn_override=default_attn):
        model = model.clone()

        transformer_options = {**model.model_options.get('transformer_options', {})}
        model.model_options['transformer_options'] = transformer_options

        transformer_options['RAVE'] = {
            "grid_size": grid_size,
            "seed": seed,
        }

        return (model, )