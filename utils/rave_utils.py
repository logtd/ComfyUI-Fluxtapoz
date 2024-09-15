import torch
import random


def shuffle_indices(size, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = list(range(size))
    random.shuffle(indices)
    return indices


def shuffle_tensors2(tensor, current_indices, target_indices):
    tensor_dict = {current_idx: t for current_idx,
                   t in zip(current_indices, tensor)}
    shuffled_tensors = [tensor_dict[current_idx]
                        for current_idx in target_indices]
    return torch.stack(shuffled_tensors)


def grid_to_list(tensor, grid_size):
    frame_count = len(tensor) * grid_size * grid_size
    flattened_list = [flatten_grid(grid.unsqueeze(
        0), [grid_size, grid_size]) for grid in tensor]
    list_tensor = torch.cat(flattened_list, dim=-2)
    return torch.cat(torch.chunk(list_tensor, frame_count, dim=-2), dim=0)


def list_to_grid(tensor, grid_size):
    grid_frame_count = grid_size * grid_size
    grid_count = len(tensor) // grid_frame_count
    flat_grids = [torch.cat([a for a in tensor[i * grid_frame_count:(i + 1)
                            * grid_frame_count]], dim=-2).unsqueeze(0) for i in range(grid_count)]
    unflattened_grids = [unflatten_grid(
        flat_grid, [grid_size, grid_size]) for flat_grid in flat_grids]
    return torch.cat(unflattened_grids, dim=0)


def flatten_grid(x, grid_shape):
    B, H, W, C = x.size()
    hs, ws = grid_shape
    img_h = H // hs
    flattened = torch.cat(torch.split(x, img_h, dim=1), dim=2)
    return flattened


def unflatten_grid(x, grid_shape):
    ''' 
    x: B x C x H x W
    '''
    B, H, W, C = x.size()
    hs, ws = grid_shape
    img_w = W // (ws)

    unflattened = torch.cat(torch.split(x, img_w, dim=2), dim=1)

    return unflattened