import torch
from typing import Tuple


def get_anchor_free_reference_points(map_size: Tuple[int, int], normalized: bool = True):
    
    w_ref_points = torch.linspace(0.5, map_size[0] - 0.5, map_size[0])
    h_ref_points = torch.linspace(0.5, map_size[1] - 0.5, map_size[1])
    if normalized:
        w_ref_points /= map_size[0]
        h_ref_points /= map_size[1]
    
    w_ref_points, h_ref_points = torch.meshgrid(w_ref_points, h_ref_points, indexing='xy')
    ref_points = torch.stack((w_ref_points, h_ref_points), dim=-1)
    return ref_points  
