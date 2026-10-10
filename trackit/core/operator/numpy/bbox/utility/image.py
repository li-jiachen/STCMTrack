import numpy as np
from ..intersection import bbox_has_intersection


def get_image_center_point(image_size: np.ndarray):
    
    return image_size / 2.


def is_bbox_intersecting_image(bbox: np.ndarray, image_size: np.ndarray):
    
    image_bbox = np.zeros((*image_size.shape[:-1], 4), dtype=image_size.dtype)
    image_bbox[..., 2:4] = image_size

    return bbox_has_intersection(bbox, image_bbox)


def bbox_clip_to_image_boundary_(bbox: np.ndarray, image_size: np.ndarray):
    
    bbox[..., 0] = np.clip(bbox[..., 0], 0, image_size[..., 0])
    bbox[..., 1] = np.clip(bbox[..., 1], 0, image_size[..., 1])
    bbox[..., 2] = np.clip(bbox[..., 2], 0, image_size[..., 0])
    bbox[..., 3] = np.clip(bbox[..., 3], 0, image_size[..., 1])
    return bbox


def bbox_clip_to_image_boundary(bbox: np.ndarray, image_size: np.ndarray):
    
    return bbox_clip_to_image_boundary_(bbox.copy(), image_size)

