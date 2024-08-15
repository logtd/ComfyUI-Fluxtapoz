from .nodes.flip_sigmas_node import InFluxFlipSigmasNode
from .nodes.influx_model_pred_node import InFluxModelSamplingPredNode, OutFluxModelSamplingPredNode
from .nodes.flux_deguidance_node import FluxDeGuidance
from .nodes.inverse_sampler_node import FluxInverseSamplerNode


NODE_CLASS_MAPPINGS = {
    "InFluxFlipSigmas": InFluxFlipSigmasNode,
    "InFluxModelSamplingPred": InFluxModelSamplingPredNode,
    "OutFluxModelSamplingPred": OutFluxModelSamplingPredNode,
    "FluxDeGuidance": FluxDeGuidance,
    "FluxInverseSampler": FluxInverseSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InFluxFlipSigmas": "Flip Flux Sigmas",
    "InFluxModelSamplingPred": "Inverse Flux Model Pred",
    "OutFluxModelSamplingPred": "Outverse Flux Model Pred",
    "FluxDeGuidance": "Flux DeGuidance",
    "FluxInverseSampler": "Flux Inverse Sampler",
}
