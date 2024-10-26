
class InFluxFlipSigmasNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "sigmas": ("SIGMAS",),
                              }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "flip"

    CATEGORY = "fluxtapoz"

    def flip(self, sigmas):
        sigmas = sigmas.flip()
        if sigmas[0] == 0:
            sigmas[0] = 1e-3
        return (sigmas, )
