import comfy.sd
import comfy.model_sampling
import comfy.latent_formats
import nodes


class InverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class InFluxModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                              "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                              "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "fluxtapoz"

    def patch(self, model, max_shift, base_shift, width, height):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = InverseCONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )


class OutCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


class ReverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output # model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


class OutFluxModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                              "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                              "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "reverse_ode": ("BOOLEAN", {"default": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "fluxtapoz"

    def patch(self, model, max_shift, base_shift, width, height, reverse_ode=False):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        if reverse_ode:
            sampling_type = ReverseCONST
        else:
            sampling_type = OutCONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )
