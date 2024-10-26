import node_helpers

class FluxDeGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "guidance": ("FLOAT", {"default": 3.5, "min": -100.0, "max": 100.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "fluxtapoz"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )