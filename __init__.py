from .nodes.flip_sigmas_node import InFluxFlipSigmasNode
from .nodes.influx_model_pred_node import InFluxModelSamplingPredNode, OutFluxModelSamplingPredNode
from .nodes.flux_deguidance_node import FluxDeGuidance
from .nodes.inverse_sampler_node import FluxInverseSamplerNode
from .nodes.configure_modified_flux_node import ConfigureModifiedFluxNode
from .nodes.mix_noise_node import FluxNoiseMixerNode
from .nodes.rectified_sampler_nodes import FluxForwardODESamplerNode, FluxReverseODESamplerNode
from .nodes.rf_edit_sampler_nodes import FlowEditForwardSamplerNode, FlowEditReverseSamplerNode, PrepareAttnBankNode, RFSingleBlocksOverrideNode, RFDoubleBlocksOverrideNode


NODE_CLASS_MAPPINGS = {
    "InFluxFlipSigmas": InFluxFlipSigmasNode,
    "InFluxModelSamplingPred": InFluxModelSamplingPredNode,
    "OutFluxModelSamplingPred": OutFluxModelSamplingPredNode,
    "FluxDeGuidance": FluxDeGuidance,
    "FluxInverseSampler": FluxInverseSamplerNode,
    "ConfigureModifiedFlux": ConfigureModifiedFluxNode,
    "FluxNoiseMixer": FluxNoiseMixerNode,
    "FluxForwardODESampler": FluxForwardODESamplerNode,
    "FluxReverseODESampler": FluxReverseODESamplerNode,
    # "AddFluxFlow": AddFluxFlowNode,
    # "ApplyFluxRaveAttention": ApplyFluxRaveAttentionNode,
    "FlowEditForwardSampler": FlowEditForwardSamplerNode,
    "FlowEditReverseSampler": FlowEditReverseSamplerNode,
    "PrepareAttnBank": PrepareAttnBankNode,
    "RFSingleBlocksOverride": RFSingleBlocksOverrideNode,
    "RFDoubleBlocksOverride": RFDoubleBlocksOverrideNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InFluxFlipSigmas": "Flip Flux Sigmas",
    "InFluxModelSamplingPred": "Inverse Flux Model Pred",
    "OutFluxModelSamplingPred": "Outverse Flux Model Pred",
    "FluxDeGuidance": "Flux DeGuidance",
    "FluxInverseSampler": "Flux Inverse Sampler",
    "ConfigureModifiedFlux": "Configure Modified Flux",
    "FluxNoiseMixer": "Flux Mix Noise",
    "FluxForwardODESampler": "Flux Forward ODE Sampler",
    "FluxReverseODESampler": "Flux Reverse ODE Sampler",
    # "AddFluxFlow": "Add Flux Flow",
    # "ApplyFluxRaveAttention": "Apply Flux Rave Attn",
    "FlowEditForwardSampler": "RF-Edit Forward Sampler",
    "FlowEditReverseSampler": "RF-Edit Reverse Sampler",
    "PrepareAttnBank": "RF-Edit Prep Attn Inj",
    "RFSingleBlocksOverride": "RF-Edit Single Layers Override",
    "RFDoubleBlocksOverride": "RF-Edit Double Layers Override"
}
