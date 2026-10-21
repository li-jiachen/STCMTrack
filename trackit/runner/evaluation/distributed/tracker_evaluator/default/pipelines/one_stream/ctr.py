
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from trackit.core.operator.numpy.bbox.area import bbox_compute_area
from trackit.core.operator.numpy.bbox.format import bbox_get_center_point, bbox_get_width_and_height
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid


@dataclass
class CTRConfig:
    enabled: bool = False
    
    mcc_enabled: bool = True  
    rgtc_enabled: bool = True  
    confidence_threshold: float = 0.40  
    
    orb_features: int = 800
    max_matches: int = 80  
    min_matches: int = 8  
    min_inlier_ratio: float = 0.25
    ransac_reproj_threshold: float = 2.0  
    max_center_shift_ratio: float = 1.0  
    
    residual_mad_scale: float = 4.4478  
    mog2_history: int = 80
    mog2_var_threshold: float = 24.0
    mog2_detect_shadows: bool = True  
    
    min_candidate_area: float = 4.0
    min_candidate_area_ratio: float = 0.05
    max_candidate_area_ratio: float = 25.0
    print_summary: bool = True

    @classmethod
    def from_dict(cls, config: dict):
        valid_names = {field.name for field in fields(cls)}
        unknown = set(config) - valid_names
        if unknown:
            raise ValueError(f'Unknown CTR settings: {sorted(unknown)}')
        return cls(**config)


@dataclass
class _TrackState:
    previous_gray: Optional[np.ndarray]
    previous_bbox: Optional[np.ndarray]
    mog2: cv2.BackgroundSubtractorMOG2
    geometric_bbox: Optional[np.ndarray] = None  
    homography: Optional[np.ndarray] = None  
    mog2_updated: bool = False


@dataclass
class _CTRStats:
    tracks: int = 0
    frames: int = 0
    homography_attempts: int = 0
    homography_successes: int = 0
    mcc_applied: int = 0
    confidence_count: int = 0
    confidence_sum: float = 0.0
    confidence_min: float = float('inf')
    confidence_max: float = -float('inf')
    low_confidence_frames: int = 0
    residual_masks: int = 0
    candidate_frames: int = 0
    candidates: int = 0
    corrections: int = 0


def build_ctr_module(config: Optional[dict]):
    if config is None:
        return None
    ctr_config = CTRConfig.from_dict(config)
    if not ctr_config.enabled or not (ctr_config.mcc_enabled or ctr_config.rgtc_enabled):
        return None
    return ConfidenceTriggeredRelocalization(ctr_config)


class ConfidenceTriggeredRelocalization:
    def __init__(self, config: CTRConfig):
        self.config = config
        self._states: Dict[int, _TrackState] = {}
        self._orb = cv2.ORB_create(nfeatures=config.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._stats = _CTRStats()

    
    def reset_statistics(self):
        self._stats = _CTRStats()

    def reset(self, task_id: int, image: Optional[torch.Tensor], bbox: np.ndarray):
        gray = _to_gray_uint8(image) if image is not None else None
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog2_history,
            varThreshold=self.config.mog2_var_threshold,
            detectShadows=self.config.mog2_detect_shadows)
        if gray is not None:
            mog2.apply(gray)
        self._states[task_id] = _TrackState(gray, bbox.astype(np.float64).copy(), mog2)
        self._stats.tracks += 1

    def forget(self, task_id: int):
        self._states.pop(task_id, None)

    def clear(self):
        self._states.clear()

    def update(self, task_id: int, image: torch.Tensor, bbox: np.ndarray, image_size: np.ndarray):
        
        state = self._states.get(task_id)
        if state is None:
            self.reset(task_id, image, bbox)
            return

        gray = _to_gray_uint8(image)
        bbox = bbox_clip_to_image_boundary(bbox.astype(np.float64).copy(), image_size)
        if bbox_is_valid(bbox):
            state.previous_bbox = bbox
        state.previous_gray = gray
        if not state.mog2_updated:
            state.mog2.apply(gray)
        state.mog2_updated = False
        state.geometric_bbox = None
        state.homography = None

    
    def compensate_search_bbox(self, task_id: int, image: torch.Tensor, image_size: np.ndarray) -> Optional[np.ndarray]:
        
        self._stats.frames += 1
        state = self._states.get(task_id)
        if state is None or state.previous_gray is None or state.previous_bbox is None:
            return None

        current_gray = _to_gray_uint8(image)
        state.geometric_bbox = None
        state.homography = None
        self._stats.homography_attempts += 1

        homography = self._estimate_homography(state.previous_gray, current_gray)
        if homography is None:
            return None
        
        state.homography = homography.copy()
        self._stats.homography_successes += 1
        geometric_bbox = self._map_previous_box(homography, state.previous_bbox, image_size)
        if geometric_bbox is None:
            return None

        state.geometric_bbox = geometric_bbox.copy()
        if not self.config.mcc_enabled:
            return None
        self._stats.mcc_applied += 1
        return geometric_bbox

    def _estimate_homography(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> Optional[np.ndarray]:
        previous_keypoints, previous_descriptors = self._orb.detectAndCompute(previous_gray, None)
        current_keypoints, current_descriptors = self._orb.detectAndCompute(current_gray, None)
        if previous_descriptors is None or current_descriptors is None:
            return None
        if len(previous_descriptors) == 0 or len(current_descriptors) == 0:
            return None

        matches = self._matcher.match(previous_descriptors, current_descriptors)
        if len(matches) < self.config.min_matches:
            return None
        matches = sorted(matches, key=lambda match: match.distance)[:self.config.max_matches]
        if len(matches) < self.config.min_matches:
            return None

        previous_points = np.float32([previous_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        current_points = np.float32([current_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            previous_points, current_points, cv2.RANSAC, self.config.ransac_reproj_threshold)
        if homography is None or inlier_mask is None:
            return None
        if not np.all(np.isfinite(homography)):
            return None

        inliers = int(inlier_mask.sum())
        if inliers < self.config.min_matches or inliers / len(matches) < self.config.min_inlier_ratio:
            return None
        return homography

    def _map_previous_box(self, homography: np.ndarray, previous_bbox: np.ndarray,
                          image_size: np.ndarray) -> Optional[np.ndarray]:
        
        if homography is None or not np.all(np.isfinite(homography)):
            return None
        previous_center = bbox_get_center_point(previous_bbox)
        try:
            projected_center = cv2.perspectiveTransform(
                previous_center.reshape(1, 1, 2).astype(np.float32),
                homography).reshape(2).astype(np.float64)
        except cv2.error:
            return None
        if not np.all(np.isfinite(projected_center)):
            return None
        if not _point_inside_image(projected_center, image_size):
            return None

        image_diagonal = max(float(np.linalg.norm(image_size.astype(np.float64))), 1.0)
        max_center_shift = image_diagonal * max(float(self.config.max_center_shift_ratio), 0.0)
        if np.linalg.norm(projected_center - previous_center) > max_center_shift:
            return None

        wh = bbox_get_width_and_height(previous_bbox)
        geometric_bbox = np.concatenate((projected_center - wh * 0.5, projected_center + wh * 0.5))
        if not bbox_is_valid(geometric_bbox):
            return None
        return geometric_bbox

    
    def correct_prediction(self, task_id: int, image: torch.Tensor, predicted_bbox: np.ndarray,
                           predicted_score: float, image_size: np.ndarray) -> Tuple[np.ndarray, bool]:
        
        self._observe_confidence(predicted_score)
        state = self._states.get(task_id)
        if state is None or not self.config.rgtc_enabled:
            return predicted_bbox, False
        if predicted_score >= float(self.config.confidence_threshold):
            return predicted_bbox, False

        self._stats.low_confidence_frames += 1
        
        reference_bbox = state.geometric_bbox if state.geometric_bbox is not None else predicted_bbox
        foreground_mask = self._foreground_mask(state, _to_gray_uint8(image))
        candidates = self._extract_candidates(foreground_mask, state.previous_bbox, image_size)
        self._stats.candidates += len(candidates)
        if len(candidates) == 0:
            return predicted_bbox, False
        self._stats.candidate_frames += 1

        selected_bbox, selected_score = self._select_candidate(
            candidates, reference_bbox, state.previous_bbox, image_size)
        if selected_bbox is None or not np.isfinite(selected_score):
            return predicted_bbox, False
        self._stats.corrections += 1
        return selected_bbox, True

    def _foreground_mask(self, state: _TrackState, gray: np.ndarray) -> np.ndarray:
        residual_mask = self._residual_mask(state, gray)
        if residual_mask is not None:
            self._stats.residual_masks += 1
        
        mog2_mask = self._mog2_mask(state, gray)
        if residual_mask is None:
            return mog2_mask
        return cv2.bitwise_or(mog2_mask, residual_mask)

    def _mog2_mask(self, state: _TrackState, gray: np.ndarray) -> np.ndarray:
        foreground_mask = state.mog2.apply(gray)
        state.mog2_updated = True
        foreground_mask = (foreground_mask == 255).astype(np.uint8) * 255  
        return foreground_mask

    def _residual_mask(self, state: _TrackState, gray: np.ndarray) -> Optional[np.ndarray]:
        
        if state.previous_gray is None or state.homography is None:
            return None
        if state.previous_gray.shape != gray.shape:
            return None

        height, width = gray.shape[:2]
        aligned_previous = cv2.warpPerspective(
            state.previous_gray, state.homography, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_region = cv2.warpPerspective(
            np.ones_like(state.previous_gray, dtype=np.float32), state.homography, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= (1.0 - 1e-6)
        if not np.any(valid_region):
            return None

        residual = cv2.absdiff(gray, aligned_previous)  
        valid_residual = residual[valid_region].astype(np.float32)
        median = float(np.median(valid_residual))
        mad = float(np.median(np.abs(valid_residual - median)))
        threshold = median + self.config.residual_mad_scale * mad  
        return ((residual > threshold) & valid_region).astype(np.uint8) * 255  

    def _extract_candidates(self, foreground_mask: np.ndarray, previous_bbox: Optional[np.ndarray],
                            image_size: np.ndarray) -> List[np.ndarray]:
        
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if previous_bbox is not None and bbox_is_valid(previous_bbox):
            previous_area = max(float(bbox_compute_area(previous_bbox)), 1.0)
        else:
            previous_area = 1.0

        min_area = max(self.config.min_candidate_area, previous_area * self.config.min_candidate_area_ratio)
        max_area = max(min_area, previous_area * self.config.max_candidate_area_ratio)

        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            bbox = np.asarray((x, y, x + w, y + h), dtype=np.float64)
            bbox = bbox_clip_to_image_boundary(bbox, image_size)
            if bbox_is_valid(bbox):
                candidates.append(bbox)
        return candidates

    def _select_candidate(self, candidates: List[np.ndarray], reference_bbox: np.ndarray,
                          previous_bbox: Optional[np.ndarray], image_size: np.ndarray
                          ) -> Tuple[Optional[np.ndarray], float]:
        
        
        reference_center = bbox_get_center_point(reference_bbox)
        if not candidates:
            return None, float('inf')
        best = min(candidates, key=lambda box: (
            float(np.linalg.norm(bbox_get_center_point(box) - reference_center)), *box.tolist()))
        return best, float(np.linalg.norm(bbox_get_center_point(best) - reference_center))

    
    def _observe_confidence(self, predicted_score: float):
        predicted_score = float(predicted_score)
        stats = self._stats
        stats.confidence_count += 1
        stats.confidence_sum += predicted_score
        stats.confidence_min = min(stats.confidence_min, predicted_score)
        stats.confidence_max = max(stats.confidence_max, predicted_score)

    def format_summary(self):
        if not self.config.print_summary:
            return []
        stats = self._stats
        if stats.confidence_count > 0:
            confidence = (f'mean={stats.confidence_sum / stats.confidence_count:.4f} '
                          f'min={stats.confidence_min:.4f} max={stats.confidence_max:.4f}')
        else:
            confidence = 'n/a'
        return [
            'CTR summary:',
            f'  tracks={stats.tracks} frames={stats.frames} mcc={self.config.mcc_enabled} '
            f'rgtc={self.config.rgtc_enabled} tau={self.config.confidence_threshold:.2f} '
            f'foreground_mask=mog2_residual_union alpha={self.config.residual_mad_scale:.4f}',
            f'  homography={stats.homography_successes}/{stats.homography_attempts} '
            f'({_safe_ratio(stats.homography_successes, stats.homography_attempts):.2%}) '
            f'mcc_applied={stats.mcc_applied}',
            f'  confidence {confidence} low_confidence={stats.low_confidence_frames}/{stats.confidence_count} '
            f'({_safe_ratio(stats.low_confidence_frames, stats.confidence_count):.2%})',
            f'  rgtc residual_masks={stats.residual_masks} candidate_frames={stats.candidate_frames} '
            f'candidates={stats.candidates} corrections={stats.corrections} '
            f'({_safe_ratio(stats.corrections, stats.low_confidence_frames):.2%} of low-confidence frames)',
        ]


def _to_gray_uint8(image: torch.Tensor) -> np.ndarray:
    
    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.numpy()
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)


def _point_inside_image(point: np.ndarray, image_size: np.ndarray) -> bool:
    if not np.all(np.isfinite(point)):
        return False
    width, height = image_size.astype(np.float64)
    return 0.0 <= float(point[0]) <= width and 0.0 <= float(point[1]) <= height


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
