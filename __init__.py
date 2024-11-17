## Inversion
from .nodes.flip_sigmas_node import InFluxFlipSigmasNode
from .nodes.influx_model_pred_node import InFluxModelSamplingPredNode, OutFluxModelSamplingPredNode
from .nodes.flux_deguidance_node import FluxDeGuidance
from .nodes.inverse_sampler_node import FluxInverseSamplerNode
from .nodes.configure_modified_flux_node import ConfigureModifiedFluxNode
from .nodes.mix_noise_node import FluxNoiseMixerNode
from .nodes.rectified_sampler_nodes import FluxForwardODESamplerNode, FluxReverseODESamplerNode
from .nodes.rf_edit_sampler_nodes import FlowEditForwardSamplerNode, FlowEditReverseSamplerNode, PrepareAttnBankNode, RFSingleBlocksOverrideNode, RFDoubleBlocksOverrideNode
## Enhance
# from .nodes.attn_override_node import FluxAttnOverrideNode
# from .nodes.apply_seg_node import SEGAttentionNode
# from .nodes.apply_pag_node import PAGAttentionNode
## Temporal
# from .nodes.apply_rave_node import ApplyFluxRaveAttentionNode
# from .nodes.add_flow_flux_node import AddFluxFlowNode
## Reference-Only
# from .nodes.apply_ref_flux import ApplyRefFluxNode
## Regional


NODE_CLASS_MAPPINGS = {
    "ConfigureModifiedFlux": ConfigureModifiedFluxNode,
    ## Inversion
    "InFluxFlipSigmas": InFluxFlipSigmasNode,
    "InFluxModelSamplingPred": InFluxModelSamplingPredNode,
    "OutFluxModelSamplingPred": OutFluxModelSamplingPredNode,
    "FluxDeGuidance": FluxDeGuidance,
    "FluxInverseSampler": FluxInverseSamplerNode,
    "FluxNoiseMixer": FluxNoiseMixerNode,
    ### RF-Inversion
    "FluxForwardODESampler": FluxForwardODESamplerNode,
    "FluxReverseODESampler": FluxReverseODESamplerNode,
    ### RF-Edit
    "FlowEditForwardSampler": FlowEditForwardSamplerNode,
    "FlowEditReverseSampler": FlowEditReverseSamplerNode,
    "PrepareAttnBank": PrepareAttnBankNode,
    "RFSingleBlocksOverride": RFSingleBlocksOverrideNode,
    "RFDoubleBlocksOverride": RFDoubleBlocksOverrideNode,
    ## ATTN
    # "FluxAttnOverride": FluxAttnOverrideNode,
    # "SEGAttention": SEGAttentionNode,
    # "PAGAttention": PAGAttentionNode,
    ## Temporal
    # "AddFluxFlow": AddFluxFlowNode,
    # "ApplyFluxRaveAttention": ApplyFluxRaveAttentionNode,
    ## Reference
    # "ApplyRefFlux": ApplyRefFluxNode,
    ## Regional
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigureModifiedFlux": "Configure Modified Flux",
    ## Inversion
    "InFluxFlipSigmas": "Flip Flux Sigmas",
    "InFluxModelSamplingPred": "Inverse Flux Model Pred",
    "OutFluxModelSamplingPred": "Outverse Flux Model Pred",
    "FluxDeGuidance": "Flux DeGuidance",
    "FluxInverseSampler": "Flux Inverse Sampler",
    "FluxNoiseMixer": "Flux Mix Noise",
    ### RF-Inversion
    "FluxForwardODESampler": "Flux Forward ODE Sampler",
    "FluxReverseODESampler": "Flux Reverse ODE Sampler",
    ### RF-Edit
    "FlowEditForwardSampler": "RF-Edit Forward Sampler",
    "FlowEditReverseSampler": "RF-Edit Reverse Sampler",
    "PrepareAttnBank": "RF-Edit Prep Attn Inj",
    "RFSingleBlocksOverride": "RF-Edit Single Layers Override",
    "RFDoubleBlocksOverride": "RF-Edit Double Layers Override",
    ## Enhance
    # "FluxAttnOverride": "Flux Attention Override",
    # "SEGAttention": "Apply DiT SEG Attention",
    # "PAGAttention": "Apply DiT PAG Attention",
    ## Temporal
    # "AddFluxFlow": "Add Flux Flow",
    # "ApplyFluxRaveAttention": "Apply Flux Rave Attn",
    ## Reference
    # "ApplyRefFlux": "Apply Reference-Only",
    ## Regional
}

