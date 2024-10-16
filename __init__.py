from .nodes.flip_sigmas_node import InFluxFlipSigmasNode
from .nodes.influx_model_pred_node import InFluxModelSamplingPredNode, OutFluxModelSamplingPredNode
from .nodes.flux_deguidance_node import FluxDeGuidance
from .nodes.inverse_sampler_node import FluxInverseSamplerNode
from .nodes.apply_ref_flux import ApplyRefFluxNode, ConfigureRefFluxNode
from .nodes.mix_noise_node import FluxNoiseMixerNode
from .nodes.rectified_sampler_nodes import FluxForwardODESamplerNode, FluxReverseODESamplerNode



NODE_CLASS_MAPPINGS = {
    "InFluxFlipSigmas": InFluxFlipSigmasNode,
    "InFluxModelSamplingPred": InFluxModelSamplingPredNode,
    "OutFluxModelSamplingPred": OutFluxModelSamplingPredNode,
    "FluxDeGuidance": FluxDeGuidance,
    "FluxInverseSampler": FluxInverseSamplerNode,
    "ApplyRefFlux": ApplyRefFluxNode,
    "ConfigureRefFlux": ConfigureRefFluxNode,
    "FluxNoiseMixer": FluxNoiseMixerNode,
    "FluxForwardODESampler": FluxForwardODESamplerNode,
    "FluxReverseODESampler": FluxReverseODESamplerNode,
    # "AddFluxFlow": AddFluxFlowNode,
    # "ApplyFluxRaveAttention": ApplyFluxRaveAttentionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InFluxFlipSigmas": "Flip Flux Sigmas",
    "InFluxModelSamplingPred": "Inverse Flux Model Pred",
    "OutFluxModelSamplingPred": "Outverse Flux Model Pred",
    "FluxDeGuidance": "Flux DeGuidance",
    "FluxInverseSampler": "Flux Inverse Sampler",
    "ApplyRefFlux": "Apply Ref Flux Model",
    "ConfigureRefFlux": "Configure Ref for Flux",
    "FluxNoiseMixer": "Flux Mix Noise",
    "FluxForwardODESampler": "Flux Forward ODE Sampler",
    "FluxReverseODESampler": "Flux Reverse ODE Sampler",
    # "AddFluxFlow": "Add Flux Flow",
    # "ApplyFluxRaveAttention": "Apply Flux Rave Attn",
}
