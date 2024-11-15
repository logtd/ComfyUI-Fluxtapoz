from einops import rearrange
import torch
import torch.nn.functional as F


def flow_attention(img_attn, num_heads, dim_head, transformer_options):
    flow = transformer_options.get('FLOW', None)
    if flow is None:
        return img_attn
    
    hidden_states = img_attn
    cond_list = transformer_options['cond_or_uncond']
    batch_size = len(hidden_states) // len(cond_list)
    flow = None
    flows = []
    flows = transformer_options['FLOW']['forward_flows']

    for possible_flow in flows:
        if possible_flow['forward_trajectory'].shape[2] == hidden_states.shape[1]:
            flow = possible_flow
            break

    if not flow:
        return hidden_states

    backward_trajectory = flow['backward_trajectory'].to(hidden_states.device)
    forward_trajectory = flow['forward_trajectory'].to(hidden_states.device)
    attn_mask = flow['attn_masks'].to(hidden_states.device)

    hidden_states = rearrange(hidden_states, "(b f) d c -> f (b c) d", f=batch_size)

    hidden_states = torch.gather(hidden_states, 2, forward_trajectory.expand(-1,hidden_states.shape[1],-1))

    hidden_states = rearrange(hidden_states, "f (b c) d -> (b d) f c", b=len(cond_list))
    hidden_states = hidden_states.view(-1, batch_size, num_heads, dim_head).transpose(1, 2).detach()

    hidden_states = F.scaled_dot_product_attention(
        hidden_states, hidden_states, hidden_states, 
        attn_mask = (attn_mask.repeat(len(cond_list),1,1,1))
    )

    hidden_states = rearrange(hidden_states, "(b d) h f c -> f (b h c) d", b=len(cond_list))
    hidden_states = torch.gather(hidden_states, 2, backward_trajectory.expand(-1,hidden_states.shape[1],-1)).detach()
    hidden_states = rearrange(hidden_states, "f (b h c) d -> (b f) h d c", b=len(cond_list), h=num_heads)
    hidden_states = rearrange(hidden_states, 'b d q f -> b q (d f)')
    
    return hidden_states