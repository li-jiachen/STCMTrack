import torch


def bbox_scale(bbox: torch.Tensor, scale: torch.Tensor):
    
    out_bbox = torch.empty_like(bbox)
    out_bbox[..., ::2] = bbox[..., ::2] * scale[..., (0, )]
    out_bbox[..., 1::2] = bbox[..., 1::2] * scale[..., (1, )]
    return out_bbox
