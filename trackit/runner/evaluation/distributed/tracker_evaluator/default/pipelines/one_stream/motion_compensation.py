from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from trackit.core.operator.numpy.bbox.area import bbox_compute_area
from trackit.core.operator.numpy.bbox.format import bbox_get_center_point, bbox_get_width_and_height
from trackit.core.operator.numpy.bbox.iou import bbox_compute_iou
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params


@dataclass
class MotionCompensationConfig:
    enabled: bool = False
    geometric_compensation_enabled: bool = True
    foreground_correction_enabled: bool = True
    confidence_threshold: float = 0.35
    confidence_threshold_mode: str = 'fixed'
    adaptive_confidence_quantile: float = 0.20
    adaptive_confidence_history: int = 50
    adaptive_confidence_warmup: int = 5
    adaptive_confidence_min_threshold: float = 0.0
    adaptive_confidence_max_threshold: float = 1.0
    ransac_reproj_threshold: float = 2.0
    min_matches: int = 8
    max_matches: int = 80
    min_inlier_ratio: float = 0.25
    orb_features: int = 800
    max_center_shift_ratio: float = 1.0
    lk_fallback_enabled: bool = True
    lk_max_corners: int = 40
    lk_quality_level: float = 0.01
    lk_min_distance: int = 3
    lk_block_size: int = 7
    lk_win_size: int = 15
    lk_max_level: int = 2
    lk_criteria_count: int = 20
    lk_criteria_eps: float = 0.03
    lk_min_points: int = 4
    lk_roi_expand_ratio: float = 2.0
    lk_max_displacement_ratio: float = 1.0
    mog2_history: int = 80
    mog2_var_threshold: float = 24.0
    mog2_detect_shadows: bool = True
    foreground_mask_mode: str = 'mog2'
    residual_threshold_k: float = 3.0
    residual_min_threshold: float = 6.0
    residual_max_threshold: float = 64.0
    residual_fallback_to_mog2: bool = False
    residual_morph_opening: bool = False
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    min_candidate_area: float = 4.0
    min_candidate_area_ratio: float = 0.05
    max_candidate_area_ratio: float = 25.0
    candidate_distance_weight: float = 0.45
    candidate_area_weight: float = 0.25
    candidate_aspect_weight: float = 0.10
    candidate_residual_weight: float = 0.15
    candidate_prior_weight: float = 0.05
    candidate_score_threshold: float = 0.1
    candidate_residual_min_support: float = 0.0
    candidate_min_distance_score: float = 0.30
    candidate_min_area_score: float = 0.20
    conservative_replacement_enabled: bool = True
    tracker_verification_enabled: bool = False
    tracker_response_weight: float = 0.25
    tracker_response_min_ratio: float = 0.60
    tracker_response_min_score: float = 0.05
    tracker_response_neighborhood: int = 1
    tracker_box_consistency_weight: float = 0.10
    tracker_box_min_iou: float = 0.0
    tracker_verified_output_mode: str = 'foreground'
    print_summary: bool = True

    @classmethod
    def from_dict(cls, config: dict):
        valid_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in config.items() if key in valid_names})


@dataclass
class _TrackState:
    previous_gray: Optional[np.ndarray]
    previous_bbox: Optional[np.ndarray]
    mog2: cv2.BackgroundSubtractorMOG2
    compensated_bbox: Optional[np.ndarray] = None
    last_homography: Optional[np.ndarray] = None
    mog_updated: bool = False
    confidence_history: list[float] = field(default_factory=list)


@dataclass
class _MotionCompensationStats:
    reset_tracks: int = 0
    tracking_frames: int = 0
    homography_attempts: int = 0
    homography_successes: int = 0
    homography_success_count: int = 0
    homography_failed_count: int = 0
    compensation_successes: int = 0
    lk_attempt_count: int = 0
    lk_success_count: int = 0
    lk_failed_count: int = 0
    lk_used_count: int = 0
    compensation_none_count: int = 0
    average_lk_dx: float = 0.0
    average_lk_dy: float = 0.0
    average_lk_valid_points: float = 0.0
    lk_dx_sum: float = 0.0
    lk_dy_sum: float = 0.0
    lk_valid_points_sum: float = 0.0
    low_confidence_frames: int = 0
    foreground_attempts: int = 0
    foreground_candidate_frames: int = 0
    foreground_total_candidates: int = 0
    foreground_corrections: int = 0
    recovery_triggered_count: int = 0
    candidate_found_count: int = 0
    candidate_accepted_count: int = 0
    candidate_rejected_count: int = 0
    reject_by_low_score_count: int = 0
    reject_by_low_residual_count: int = 0
    reject_by_distance_count: int = 0
    reject_by_area_count: int = 0
    average_candidate_score: float = 0.0
    average_residual_score: float = 0.0
    candidate_score_sum: float = 0.0
    residual_score_sum: float = 0.0
    candidate_score_count: int = 0
    confidence_count: int = 0
    confidence_sum: float = 0.0
    confidence_min: float = float('inf')
    confidence_max: float = -float('inf')
    gate_threshold_count: int = 0
    gate_threshold_sum: float = 0.0
    gate_threshold_min: float = float('inf')
    gate_threshold_max: float = -float('inf')


@dataclass
class _TrackerVerificationContext:
    score_map: np.ndarray
    box_map: Optional[np.ndarray]
    cropping_params: np.ndarray
    search_region_size: np.ndarray
    image_size: np.ndarray
    predicted_score: float


@dataclass
class _TrackerCandidateSupport:
    response_score: float
    relative_response: float
    box_iou: float
    tracker_bbox: Optional[np.ndarray]


@dataclass
class _CandidateSelectionResult:
    bbox: Optional[np.ndarray]
    score: float
    distance_score: float = 0.0
    area_score: float = 0.0
    aspect_score: float = 0.0
    residual_score: float = 0.0
    prior_score: float = 0.0


def build_motion_compensation_module(config: Optional[dict]):
    if config is None:
        return None
    motion_compensation_config = MotionCompensationConfig.from_dict(config)
    if not motion_compensation_config.enabled:
        return None
    return MotionCompensationModule(motion_compensation_config)


class MotionCompensationModule:
    def __init__(self, config: MotionCompensationConfig):
        self.config = config
        self._states: Dict[int, _TrackState] = {}
        self._orb = cv2.ORB_create(nfeatures=config.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._stats = _MotionCompensationStats()

    def reset_statistics(self):
        self._stats = _MotionCompensationStats()

    def reset(self, task_id: int, image: Optional[torch.Tensor], bbox: np.ndarray):
        gray = _to_gray_uint8(image) if image is not None else None
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog2_history,
            varThreshold=self.config.mog2_var_threshold,
            detectShadows=self.config.mog2_detect_shadows)
        if gray is not None:
            mog2.apply(gray)
        self._states[task_id] = _TrackState(gray, bbox.astype(np.float64).copy(), mog2)
        self._stats.reset_tracks += 1

    def forget(self, task_id: int):
        self._states.pop(task_id, None)

    def clear(self):
        self._states.clear()

    def compensate_search_bbox(self, task_id: int, image: torch.Tensor, image_size: np.ndarray) -> Optional[np.ndarray]:
        self._stats.tracking_frames += 1
        if not self.config.geometric_compensation_enabled:
            return None
        state = self._states.get(task_id)
        if state is None or state.previous_gray is None or state.previous_bbox is None:
            return None

        current_gray = _to_gray_uint8(image)
        state.compensated_bbox = None
        state.last_homography = None
        self._stats.homography_attempts += 1

        homography = self._estimate_homography(state.previous_gray, current_gray)
        compensated_bbox = None
        if homography is not None:
            self._stats.homography_successes += 1
            compensated_bbox = self._compensate_bbox_with_homography(
                homography, state.previous_bbox, image_size)

        if compensated_bbox is not None:
            state.compensated_bbox = compensated_bbox.copy()
            state.last_homography = homography.copy()
            self._stats.homography_success_count += 1
            self._stats.compensation_successes += 1
            return compensated_bbox

        self._stats.homography_failed_count += 1
        if self.config.lk_fallback_enabled:
            self._stats.lk_attempt_count += 1
            lk_bbox = self._estimate_lk_flow_center(
                state.previous_gray,
                current_gray,
                state.previous_bbox,
                image_size)
            if lk_bbox is not None:
                state.compensated_bbox = lk_bbox.copy()
                self._stats.lk_success_count += 1
                self._stats.lk_used_count += 1
                self._stats.compensation_successes += 1
                self._update_lk_averages()
                return lk_bbox
            self._stats.lk_failed_count += 1

        self._stats.compensation_none_count += 1
        return None

    def correct_prediction(self, task_id: int, image: torch.Tensor, predicted_bbox: np.ndarray,
                           predicted_score: float, image_size: np.ndarray, *,
                           tracker_score_map: Optional[np.ndarray] = None,
                           tracker_box_map: Optional[np.ndarray] = None,
                           cropping_params: Optional[np.ndarray] = None,
                           search_region_size: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        self._observe_confidence(predicted_score)
        state = self._states.get(task_id)
        if state is None or not self.config.foreground_correction_enabled:
            return predicted_bbox, False
        gate_threshold = self._get_confidence_gate_threshold(state)
        self._append_confidence_history(state, predicted_score)
        if gate_threshold is None or predicted_score >= gate_threshold:
            return predicted_bbox, False

        self._stats.low_confidence_frames += 1
        self._stats.recovery_triggered_count += 1
        self._stats.foreground_attempts += 1
        reference_bbox = state.compensated_bbox if state.compensated_bbox is not None else predicted_bbox
        foreground_mask = self._foreground_mask(state, image, reference_bbox)
        candidates = self._extract_candidates(foreground_mask, state.previous_bbox, image_size)
        self._stats.foreground_total_candidates += len(candidates)
        if len(candidates) == 0:
            return predicted_bbox, False
        self._stats.foreground_candidate_frames += 1
        self._stats.candidate_found_count += 1

        tracker_verification_context = self._make_tracker_verification_context(
            tracker_score_map,
            tracker_box_map,
            cropping_params,
            search_region_size,
            image_size,
            predicted_score)

        residual_mask = None
        use_residual_scoring = (
            self.config.candidate_residual_weight > 0.0 or
            self.config.candidate_residual_min_support > 0.0
        )
        if use_residual_scoring:
            residual_foreground = self._motion_residual_foreground_mask(state, _to_gray_uint8(image))
            if residual_foreground is not None:
                residual_mask = self._postprocess_foreground_mask(
                    residual_foreground, self.config.residual_morph_opening)

        selection = self._select_candidate(
            candidates,
            reference_bbox,
            state.previous_bbox,
            predicted_bbox,
            state.compensated_bbox,
            image_size,
            residual_mask,
            self.config.candidate_distance_weight,
            self.config.candidate_area_weight,
            self.config.candidate_aspect_weight,
            self.config.candidate_residual_weight,
            self.config.candidate_prior_weight,
            tracker_verification_context,
            self.config.tracker_response_weight,
            self.config.tracker_response_min_ratio,
            self.config.tracker_response_min_score,
            self.config.tracker_response_neighborhood,
            self.config.tracker_box_consistency_weight,
            self.config.tracker_box_min_iou,
            self.config.tracker_verified_output_mode)
        self._observe_candidate_selection(selection)
        reject_reason = self._replacement_reject_reason(selection)
        if reject_reason is not None:
            self._stats.candidate_rejected_count += 1
            if reject_reason == 'low_residual':
                self._stats.reject_by_low_residual_count += 1
            elif reject_reason == 'distance':
                self._stats.reject_by_distance_count += 1
            elif reject_reason == 'area':
                self._stats.reject_by_area_count += 1
            else:
                self._stats.reject_by_low_score_count += 1
            return predicted_bbox, False
        self._stats.foreground_corrections += 1
        self._stats.candidate_accepted_count += 1
        return selection.bbox, True

    def _observe_candidate_selection(self, selection: _CandidateSelectionResult):
        if selection.bbox is None or not np.isfinite(selection.score):
            return
        stats = self._stats
        stats.candidate_score_count += 1
        stats.candidate_score_sum += float(selection.score)
        stats.residual_score_sum += float(selection.residual_score)
        stats.average_candidate_score = stats.candidate_score_sum / stats.candidate_score_count
        stats.average_residual_score = stats.residual_score_sum / stats.candidate_score_count

    def _replacement_reject_reason(self, selection: _CandidateSelectionResult) -> Optional[str]:
        if selection.bbox is None or not np.isfinite(selection.score):
            return 'low_score'
        if selection.score < float(self.config.candidate_score_threshold):
            return 'low_score'
        if not self.config.conservative_replacement_enabled:
            return None
        if selection.residual_score < float(self.config.candidate_residual_min_support):
            return 'low_residual'
        if selection.distance_score < float(self.config.candidate_min_distance_score):
            return 'distance'
        if selection.area_score < float(self.config.candidate_min_area_score):
            return 'area'
        return None

    def format_summary(self):
        if not self.config.print_summary:
            return []
        stats = self._stats
        confidence_mean = stats.confidence_sum / stats.confidence_count if stats.confidence_count > 0 else float('nan')
        confidence_min = stats.confidence_min if stats.confidence_count > 0 else float('nan')
        confidence_max = stats.confidence_max if stats.confidence_count > 0 else float('nan')
        gate_threshold_mean = (
            stats.gate_threshold_sum / stats.gate_threshold_count
            if stats.gate_threshold_count > 0 else float('nan')
        )
        gate_threshold_min = stats.gate_threshold_min if stats.gate_threshold_count > 0 else float('nan')
        gate_threshold_max = stats.gate_threshold_max if stats.gate_threshold_count > 0 else float('nan')
        homography_rate = _safe_ratio(stats.homography_successes, stats.homography_attempts)
        homography_compensation_rate = _safe_ratio(stats.homography_success_count, stats.homography_attempts)
        lk_success_rate = _safe_ratio(stats.lk_success_count, stats.lk_attempt_count)
        lk_used_rate = _safe_ratio(stats.lk_used_count, stats.homography_failed_count)
        compensation_rate = _safe_ratio(stats.compensation_successes, stats.tracking_frames)
        low_confidence_rate = _safe_ratio(stats.low_confidence_frames, stats.confidence_count)
        correction_rate = _safe_ratio(stats.foreground_corrections, stats.foreground_attempts)
        candidate_accept_rate = _safe_ratio(stats.candidate_accepted_count, stats.candidate_found_count)
        candidate_reject_rate = _safe_ratio(stats.candidate_rejected_count, stats.candidate_found_count)
        average_candidate_score = (
            stats.average_candidate_score if stats.candidate_score_count > 0 else float('nan')
        )
        average_residual_score = (
            stats.average_residual_score if stats.candidate_score_count > 0 else float('nan')
        )
        average_lk_dx = stats.average_lk_dx if stats.lk_success_count > 0 else float('nan')
        average_lk_dy = stats.average_lk_dy if stats.lk_success_count > 0 else float('nan')
        average_lk_valid_points = stats.average_lk_valid_points if stats.lk_success_count > 0 else float('nan')
        return [
            'CTR summary:',
            f'  tracks={stats.reset_tracks} frames={stats.tracking_frames} '
            f'geom_enabled={self.config.geometric_compensation_enabled} '
            f'lk_enabled={self.config.lk_fallback_enabled} '
            f'fg_enabled={self.config.foreground_correction_enabled} '
            f'fg_mode={self.config.foreground_mask_mode} '
            f'confidence_threshold={self.config.confidence_threshold:.4f} '
            f'confidence_gate={self.config.confidence_threshold_mode}',
            f'  adaptive_confidence q={self.config.adaptive_confidence_quantile:.4f} '
            f'history={self.config.adaptive_confidence_history} '
            f'warmup={self.config.adaptive_confidence_warmup} '
            f'threshold mean={gate_threshold_mean:.4f} min={gate_threshold_min:.4f} max={gate_threshold_max:.4f}',
            f'  homography={stats.homography_successes}/{stats.homography_attempts} ({homography_rate:.2%}) '
            f'compensation={stats.compensation_successes}/{stats.tracking_frames} ({compensation_rate:.2%})',
            f'  homography_success_count={stats.homography_success_count} ({homography_compensation_rate:.2%}) '
            f'homography_failed_count={stats.homography_failed_count} '
            f'compensation_none_count={stats.compensation_none_count}',
            f'  lk_attempt_count={stats.lk_attempt_count} lk_success_count={stats.lk_success_count} '
            f'({lk_success_rate:.2%}) lk_failed_count={stats.lk_failed_count} '
            f'lk_used_count={stats.lk_used_count} ({lk_used_rate:.2%}) '
            f'average_lk_dx={average_lk_dx:.4f} average_lk_dy={average_lk_dy:.4f} '
            f'average_lk_valid_points={average_lk_valid_points:.4f}',
            f'  confidence mean={confidence_mean:.4f} min={confidence_min:.4f} max={confidence_max:.4f} '
            f'low_confidence={stats.low_confidence_frames}/{stats.confidence_count} ({low_confidence_rate:.2%})',
            f'  foreground attempts={stats.foreground_attempts} candidate_frames={stats.foreground_candidate_frames} '
            f'candidates={stats.foreground_total_candidates} corrections={stats.foreground_corrections} ({correction_rate:.2%})',
            f'  recovery triggered={stats.recovery_triggered_count} candidate_found={stats.candidate_found_count} '
            f'accepted={stats.candidate_accepted_count} ({candidate_accept_rate:.2%}) '
            f'rejected={stats.candidate_rejected_count} ({candidate_reject_rate:.2%})',
            f'  reject low_score={stats.reject_by_low_score_count} '
            f'low_residual={stats.reject_by_low_residual_count} '
            f'distance={stats.reject_by_distance_count} area={stats.reject_by_area_count} '
            f'avg_candidate_score={average_candidate_score:.4f} avg_residual_score={average_residual_score:.4f}',
            f'  candidate_weights distance={self.config.candidate_distance_weight:.4f} '
            f'area={self.config.candidate_area_weight:.4f} aspect={self.config.candidate_aspect_weight:.4f} '
            f'residual={self.config.candidate_residual_weight:.4f} prior={self.config.candidate_prior_weight:.4f} '
            f'threshold={self.config.candidate_score_threshold:.4f} residual_min={self.config.candidate_residual_min_support:.4f} '
            f'min_distance={self.config.candidate_min_distance_score:.4f} min_area={self.config.candidate_min_area_score:.4f} '
            f'conservative={self.config.conservative_replacement_enabled}',
            f'  tracker_verification={self.config.tracker_verification_enabled} '
            f'response_min={self.config.tracker_response_min_score:.4f} '
            f'response_ratio={self.config.tracker_response_min_ratio:.4f} '
            f'box_min_iou={self.config.tracker_box_min_iou:.4f} '
            f'output_mode={self.config.tracker_verified_output_mode}',
        ]

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
        if not state.mog_updated:
            state.mog2.apply(gray)
        state.mog_updated = False
        state.compensated_bbox = None
        state.last_homography = None

    def _observe_confidence(self, predicted_score: float):
        predicted_score = float(predicted_score)
        self._stats.confidence_count += 1
        self._stats.confidence_sum += predicted_score
        self._stats.confidence_min = min(self._stats.confidence_min, predicted_score)
        self._stats.confidence_max = max(self._stats.confidence_max, predicted_score)

    def _get_confidence_gate_threshold(self, state: _TrackState) -> Optional[float]:
        mode = self.config.confidence_threshold_mode
        if mode == 'fixed':
            threshold = float(self.config.confidence_threshold)
        elif mode == 'adaptive_quantile':
            if len(state.confidence_history) < self.config.adaptive_confidence_warmup:
                return None
            threshold = float(np.quantile(
                np.asarray(state.confidence_history, dtype=np.float64),
                self.config.adaptive_confidence_quantile))
            threshold = float(np.clip(
                threshold,
                self.config.adaptive_confidence_min_threshold,
                self.config.adaptive_confidence_max_threshold))
        else:
            raise ValueError(f'Unknown confidence_threshold_mode: {mode}')

        self._stats.gate_threshold_count += 1
        self._stats.gate_threshold_sum += threshold
        self._stats.gate_threshold_min = min(self._stats.gate_threshold_min, threshold)
        self._stats.gate_threshold_max = max(self._stats.gate_threshold_max, threshold)
        return threshold

    def _append_confidence_history(self, state: _TrackState, predicted_score: float):
        state.confidence_history.append(float(predicted_score))
        history_size = int(self.config.adaptive_confidence_history)
        if history_size > 0 and len(state.confidence_history) > history_size:
            del state.confidence_history[:len(state.confidence_history) - history_size]

    def _make_tracker_verification_context(self, tracker_score_map: Optional[np.ndarray],
                                           tracker_box_map: Optional[np.ndarray],
                                           cropping_params: Optional[np.ndarray],
                                           search_region_size: Optional[np.ndarray],
                                           image_size: np.ndarray,
                                           predicted_score: float) -> Optional[_TrackerVerificationContext]:
        if not self.config.tracker_verification_enabled:
            return None
        if tracker_score_map is None or cropping_params is None or search_region_size is None:
            return None

        score_map = np.asarray(tracker_score_map, dtype=np.float64)
        if score_map.ndim != 2 or score_map.size == 0:
            return None
        box_map = None
        if tracker_box_map is not None:
            box_map = np.asarray(tracker_box_map, dtype=np.float64)
            if box_map.ndim != 3 or box_map.shape[:2] != score_map.shape or box_map.shape[2] != 4:
                box_map = None
        return _TrackerVerificationContext(
            score_map,
            box_map,
            np.asarray(cropping_params, dtype=np.float64),
            np.asarray(search_region_size, dtype=np.float64),
            np.asarray(image_size, dtype=np.float64),
            float(predicted_score))

    def _compensate_bbox_with_homography(self, homography: np.ndarray, previous_bbox: np.ndarray,
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
        if not self._center_inside_image(projected_center, image_size):
            return None

        image_diagonal = max(float(np.linalg.norm(image_size.astype(np.float64))), 1.0)
        max_center_shift = image_diagonal * max(float(self.config.max_center_shift_ratio), 0.0)
        if np.linalg.norm(projected_center - previous_center) > max_center_shift:
            return None

        wh = bbox_get_width_and_height(previous_bbox)
        compensated_bbox = np.concatenate((projected_center - wh * 0.5, projected_center + wh * 0.5))
        compensated_bbox = bbox_clip_to_image_boundary(compensated_bbox, image_size)
        if not bbox_is_valid(compensated_bbox):
            return None
        return compensated_bbox

    @staticmethod
    def _center_inside_image(center: np.ndarray, image_size: np.ndarray) -> bool:
        if not np.all(np.isfinite(center)):
            return False
        width, height = image_size.astype(np.float64)
        return 0.0 <= float(center[0]) <= width and 0.0 <= float(center[1]) <= height

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

    def _estimate_lk_flow_center(self, previous_gray: np.ndarray, current_gray: np.ndarray,
                                 previous_bbox: np.ndarray, image_size: np.ndarray) -> Optional[np.ndarray]:
        if previous_gray is None or current_gray is None:
            return None
        if previous_gray.shape != current_gray.shape:
            return None
        image_size = image_size.astype(np.float64)
        previous_bbox = bbox_clip_to_image_boundary(previous_bbox.astype(np.float64).copy(), image_size)
        if not bbox_is_valid(previous_bbox):
            return None

        previous_center = bbox_get_center_point(previous_bbox)
        wh = bbox_get_width_and_height(previous_bbox)
        if not np.all(np.isfinite(previous_center)) or not np.all(np.isfinite(wh)):
            return None
        if wh[0] <= 0.0 or wh[1] <= 0.0:
            return None

        height, width = previous_gray.shape[:2]
        expand_ratio = max(float(self.config.lk_roi_expand_ratio), 1.0)
        roi_half_size = np.maximum(wh * expand_ratio * 0.5, 1.0)
        roi_min = np.floor(previous_center - roi_half_size).astype(np.int64)
        roi_max = np.ceil(previous_center + roi_half_size).astype(np.int64)
        x1 = int(np.clip(roi_min[0], 0, width))
        y1 = int(np.clip(roi_min[1], 0, height))
        x2 = int(np.clip(roi_max[0], 0, width))
        y2 = int(np.clip(roi_max[1], 0, height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        roi_gray = previous_gray[y1:y2, x1:x2]
        block_size = max(1, int(self.config.lk_block_size))
        block_size = min(block_size, roi_gray.shape[0], roi_gray.shape[1])
        try:
            roi_points = cv2.goodFeaturesToTrack(
                roi_gray,
                maxCorners=max(1, int(self.config.lk_max_corners)),
                qualityLevel=max(float(self.config.lk_quality_level), 1e-6),
                minDistance=max(float(self.config.lk_min_distance), 0.0),
                blockSize=block_size)
        except cv2.error:
            return None
        if roi_points is None or len(roi_points) == 0:
            return None

        previous_points = roi_points.astype(np.float32)
        previous_points[:, 0, 0] += float(x1)
        previous_points[:, 0, 1] += float(y1)
        win_size = max(3, int(self.config.lk_win_size))
        if win_size % 2 == 0:
            win_size += 1
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max(1, int(self.config.lk_criteria_count)),
            max(float(self.config.lk_criteria_eps), 1e-6),
        )
        try:
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(win_size, win_size),
                maxLevel=max(0, int(self.config.lk_max_level)),
                criteria=criteria)
        except cv2.error:
            return None
        if current_points is None or status is None:
            return None

        old_points = previous_points.reshape(-1, 2).astype(np.float64)
        new_points = current_points.reshape(-1, 2).astype(np.float64)
        valid = status.reshape(-1) == 1
        valid &= np.all(np.isfinite(old_points), axis=1)
        valid &= np.all(np.isfinite(new_points), axis=1)
        valid &= new_points[:, 0] >= 0.0
        valid &= new_points[:, 0] <= float(width)
        valid &= new_points[:, 1] >= 0.0
        valid &= new_points[:, 1] <= float(height)
        displacements = new_points - old_points
        displacement_norms = np.linalg.norm(displacements, axis=1)
        max_displacement = max(
            float(np.linalg.norm(wh)) * max(float(self.config.lk_max_displacement_ratio), 0.0),
            1.0)
        valid &= np.isfinite(displacement_norms)
        valid &= displacement_norms <= max_displacement
        valid_displacements = displacements[valid]
        valid_points = int(valid_displacements.shape[0])
        if valid_points < max(1, int(self.config.lk_min_points)):
            return None

        median_displacement = np.median(valid_displacements, axis=0).astype(np.float64)
        if not np.all(np.isfinite(median_displacement)):
            return None
        if float(np.linalg.norm(median_displacement)) > max_displacement:
            return None

        lk_center = previous_center + median_displacement
        if not np.all(np.isfinite(lk_center)):
            return None
        lk_compensated_bbox = np.concatenate((lk_center - wh * 0.5, lk_center + wh * 0.5))
        lk_compensated_bbox = bbox_clip_to_image_boundary(lk_compensated_bbox, image_size)
        if not bbox_is_valid(lk_compensated_bbox):
            return None

        self._stats.lk_dx_sum += float(median_displacement[0])
        self._stats.lk_dy_sum += float(median_displacement[1])
        self._stats.lk_valid_points_sum += float(valid_points)
        return lk_compensated_bbox

    def _update_lk_averages(self):
        count = self._stats.lk_success_count
        if count <= 0:
            self._stats.average_lk_dx = 0.0
            self._stats.average_lk_dy = 0.0
            self._stats.average_lk_valid_points = 0.0
            return
        self._stats.average_lk_dx = self._stats.lk_dx_sum / count
        self._stats.average_lk_dy = self._stats.lk_dy_sum / count
        self._stats.average_lk_valid_points = self._stats.lk_valid_points_sum / count

    def _foreground_mask(self, state: _TrackState, image: torch.Tensor,
                         reference_bbox: Optional[np.ndarray]) -> np.ndarray:
        gray = _to_gray_uint8(image)
        mode = self.config.foreground_mask_mode
        if mode == 'mog2':
            return self._mog2_foreground_mask(state, gray)
        if mode in ('mog2_residual_union', 'residual_mog2_union', 'compensated_residual_union'):
            return self._mog2_residual_union_mask(state, gray)
        if mode in ('motion_residual', 'residual', 'compensated_residual'):
            residual_mask = self._motion_residual_foreground_mask(state, gray)
            if residual_mask is not None:
                return self._postprocess_foreground_mask(residual_mask, self.config.residual_morph_opening)
            if self.config.residual_fallback_to_mog2:
                return self._mog2_foreground_mask(state, gray)
            return np.zeros_like(gray, dtype=np.uint8)
        raise ValueError(f'Unknown foreground_mask_mode: {mode}')

    def _mog2_foreground_mask(self, state: _TrackState, gray: np.ndarray) -> np.ndarray:
        foreground_mask = state.mog2.apply(gray)
        state.mog_updated = True
        foreground_mask = (foreground_mask == 255).astype(np.uint8) * 255
        return self._postprocess_foreground_mask(foreground_mask, True)

    def _mog2_residual_union_mask(self, state: _TrackState, gray: np.ndarray) -> np.ndarray:
        mog2_mask = self._mog2_foreground_mask(state, gray)
        residual_mask = self._motion_residual_foreground_mask(state, gray)
        if residual_mask is None:
            return mog2_mask

        residual_mask = self._postprocess_foreground_mask(residual_mask, self.config.residual_morph_opening)
        union_mask = cv2.bitwise_or(mog2_mask, residual_mask)
        return self._postprocess_foreground_mask(union_mask, False)

    def _motion_residual_foreground_mask(self, state: _TrackState, gray: np.ndarray) -> Optional[np.ndarray]:
        if state.previous_gray is None or state.last_homography is None:
            return None
        if state.previous_gray.shape != gray.shape:
            return None

        height, width = gray.shape[:2]
        homography = state.last_homography
        warped_previous = cv2.warpPerspective(
            state.previous_gray,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        valid_mask = cv2.warpPerspective(
            np.full_like(state.previous_gray, 255, dtype=np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0) > 0

        if not np.any(valid_mask):
            return None

        residual = cv2.absdiff(gray, warped_previous)
        valid_residual = residual[valid_mask].astype(np.float32)
        median = float(np.median(valid_residual))
        mad = float(np.median(np.abs(valid_residual - median)))
        robust_sigma = 1.4826 * mad
        threshold = median + self.config.residual_threshold_k * robust_sigma
        threshold = float(np.clip(
            threshold,
            self.config.residual_min_threshold,
            self.config.residual_max_threshold))

        foreground_mask = ((residual >= threshold) & valid_mask).astype(np.uint8) * 255
        return foreground_mask

    def _postprocess_foreground_mask(self, foreground_mask: np.ndarray, apply_opening: bool) -> np.ndarray:
        kernel_size = max(1, int(self.config.morph_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if apply_opening:
            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)
        return foreground_mask

    def _extract_candidates(self, foreground_mask: np.ndarray, previous_bbox: Optional[np.ndarray],
                            image_size: np.ndarray):
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

    @staticmethod
    def _select_candidate(candidates, reference_bbox: np.ndarray, previous_bbox: Optional[np.ndarray],
                          predicted_bbox: Optional[np.ndarray], compensated_bbox: Optional[np.ndarray],
                          image_size: np.ndarray, residual_mask: Optional[np.ndarray] = None,
                          candidate_distance_weight: float = 0.45,
                          candidate_area_weight: float = 0.25,
                          candidate_aspect_weight: float = 0.10,
                          candidate_residual_weight: float = 0.15,
                          candidate_prior_weight: float = 0.05,
                          tracker_verification_context: Optional[_TrackerVerificationContext] = None,
                          tracker_response_weight: float = 0.0,
                          tracker_response_min_ratio: float = 0.0,
                          tracker_response_min_score: float = 0.0,
                          tracker_response_neighborhood: int = 1,
                          tracker_box_consistency_weight: float = 0.0,
                          tracker_box_min_iou: float = 0.0,
                          tracker_verified_output_mode: str = 'foreground') -> _CandidateSelectionResult:
        reference_center = bbox_get_center_point(reference_bbox)
        image_diagonal = max(float(np.linalg.norm(image_size.astype(np.float64))), 1.0)
        distance_weight = float(max(candidate_distance_weight, 0.0))
        area_weight = float(max(candidate_area_weight, 0.0))
        aspect_weight = float(max(candidate_aspect_weight, 0.0))
        residual_weight = float(max(candidate_residual_weight, 0.0))
        prior_weight = float(max(candidate_prior_weight, 0.0))
        tracker_weight = float(np.clip(tracker_response_weight, 0.0, 1.0))
        tracker_box_weight = float(np.clip(tracker_box_consistency_weight, 0.0, 1.0))

        if previous_bbox is not None and bbox_is_valid(previous_bbox):
            previous_area = max(float(bbox_compute_area(previous_bbox)), 1.0)
            previous_wh = bbox_get_width_and_height(previous_bbox)
            previous_aspect = max(float(previous_wh[0] / max(previous_wh[1], 1e-6)), 1e-6)
        else:
            previous_area = max(float(bbox_compute_area(reference_bbox)), 1.0)
            reference_wh = bbox_get_width_and_height(reference_bbox)
            previous_aspect = max(float(reference_wh[0] / max(reference_wh[1], 1e-6)), 1e-6)

        if predicted_bbox is not None and bbox_is_valid(predicted_bbox):
            predicted_center = bbox_get_center_point(predicted_bbox)
        else:
            predicted_center = reference_center

        if compensated_bbox is not None and bbox_is_valid(compensated_bbox):
            compensated_center = bbox_get_center_point(compensated_bbox)
        else:
            compensated_center = None

        best = _CandidateSelectionResult(None, -float('inf'))
        for bbox in candidates:
            output_bbox = bbox
            center = bbox_get_center_point(bbox)
            area = max(float(bbox_compute_area(bbox)), 1.0)
            wh = bbox_get_width_and_height(bbox)
            aspect = max(float(wh[0] / max(wh[1], 1e-6)), 1e-6)

            distance_score = _center_consistency_score(center, reference_center, image_diagonal)
            area_score = float(np.exp(-abs(np.log(area / previous_area))))
            aspect_score = float(np.exp(-abs(np.log(aspect / previous_aspect))))
            residual_score = _candidate_residual_support(residual_mask, bbox) if residual_mask is not None else 0.0
            predicted_prior_score = _center_consistency_score(center, predicted_center, image_diagonal)
            if compensated_center is not None:
                compensated_prior_score = _center_consistency_score(center, compensated_center, image_diagonal)
                prior_score = 0.70 * compensated_prior_score + 0.30 * predicted_prior_score
            else:
                prior_score = predicted_prior_score

            score = (
                distance_weight * distance_score +
                area_weight * area_score +
                aspect_weight * aspect_score +
                residual_weight * residual_score +
                prior_weight * prior_score
            )

            if tracker_verification_context is not None:
                support = _tracker_candidate_support(
                    tracker_verification_context,
                    bbox,
                    tracker_response_neighborhood)
                if support is None:
                    continue
                response_threshold = max(
                    float(tracker_response_min_score),
                    tracker_verification_context.predicted_score * float(tracker_response_min_ratio))
                if support.response_score < response_threshold:
                    continue
                if support.box_iou < float(tracker_box_min_iou):
                    continue
                score = _weighted_average(score, support.relative_response, tracker_weight)
                if tracker_box_weight > 0.0:
                    score = _weighted_average(score, support.box_iou, tracker_box_weight)
                if tracker_verified_output_mode == 'tracker_box' and support.tracker_bbox is not None:
                    output_bbox = support.tracker_bbox
                elif tracker_verified_output_mode != 'foreground':
                    raise ValueError(f'Unknown tracker_verified_output_mode: {tracker_verified_output_mode}')

            if score > best.score:
                best = _CandidateSelectionResult(
                    output_bbox,
                    float(score),
                    float(distance_score),
                    float(area_score),
                    float(aspect_score),
                    float(residual_score),
                    float(prior_score))
        return best


def _to_gray_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.numpy()
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max(initial=0.) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)


def _center_consistency_score(center: np.ndarray, reference_center: np.ndarray, image_diagonal: float) -> float:
    return 1.0 - min(float(np.linalg.norm(center - reference_center)) / max(float(image_diagonal), 1.0), 1.0)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _weighted_average(base_score: float, extra_score: float, extra_weight: float) -> float:
    extra_weight = float(np.clip(extra_weight, 0.0, 1.0))
    return (1.0 - extra_weight) * float(base_score) + extra_weight * float(extra_score)


def _candidate_residual_support(residual_mask: np.ndarray, bbox: np.ndarray) -> float:
    height, width = residual_mask.shape[:2]
    x1, y1, x2, y2 = np.round(bbox).astype(np.int64)
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    candidate_mask = residual_mask[y1:y2, x1:x2]
    if candidate_mask.size == 0:
        return 0.0
    return float(np.count_nonzero(candidate_mask)) / float(candidate_mask.size)


def _tracker_candidate_support(context: _TrackerVerificationContext, bbox: np.ndarray,
                               neighborhood: int) -> Optional[_TrackerCandidateSupport]:
    score_map = context.score_map
    map_height, map_width = score_map.shape
    if map_height <= 0 or map_width <= 0:
        return None

    bbox_in_search = apply_siamfc_cropping_to_boxes(bbox, context.cropping_params)
    if not bbox_is_valid(bbox_in_search):
        return None

    search_width = max(float(context.search_region_size[0]), 1.0)
    search_height = max(float(context.search_region_size[1]), 1.0)
    if bbox_in_search[2] <= 0 or bbox_in_search[0] >= search_width or \
            bbox_in_search[3] <= 0 or bbox_in_search[1] >= search_height:
        return None

    response_bbox = np.asarray((
        bbox_in_search[0] / search_width * map_width,
        bbox_in_search[1] / search_height * map_height,
        bbox_in_search[2] / search_width * map_width,
        bbox_in_search[3] / search_height * map_height), dtype=np.float64)
    neighborhood = max(0, int(neighborhood))
    x1 = int(np.floor(response_bbox[0])) - neighborhood
    y1 = int(np.floor(response_bbox[1])) - neighborhood
    x2 = int(np.ceil(response_bbox[2])) + neighborhood + 1
    y2 = int(np.ceil(response_bbox[3])) + neighborhood + 1

    if x2 <= x1 or y2 <= y1:
        center = bbox_get_center_point(bbox_in_search)
        x1 = int(np.floor(center[0] / search_width * map_width)) - neighborhood
        y1 = int(np.floor(center[1] / search_height * map_height)) - neighborhood
        x2 = x1 + 2 * neighborhood + 1
        y2 = y1 + 2 * neighborhood + 1

    x1 = int(np.clip(x1, 0, map_width))
    x2 = int(np.clip(x2, 0, map_width))
    y1 = int(np.clip(y1, 0, map_height))
    y2 = int(np.clip(y2, 0, map_height))
    if x2 <= x1 or y2 <= y1:
        return None

    local_score_map = score_map[y1:y2, x1:x2]
    if local_score_map.size == 0:
        return None
    local_index = int(np.argmax(local_score_map))
    local_y, local_x = np.unravel_index(local_index, local_score_map.shape)
    response_y = y1 + int(local_y)
    response_x = x1 + int(local_x)
    response_score = float(local_score_map[local_y, local_x])
    relative_response = response_score / max(float(context.predicted_score), 1e-6)
    relative_response = float(np.clip(relative_response, 0.0, 1.0))

    box_iou = 0.0
    tracker_bbox = None
    if context.box_map is not None:
        predicted_box_in_search = context.box_map[response_y, response_x]
        predicted_box = apply_siamfc_cropping_to_boxes(
            predicted_box_in_search,
            reverse_siamfc_cropping_params(context.cropping_params))
        predicted_box = bbox_clip_to_image_boundary(predicted_box, context.image_size)
        if bbox_is_valid(predicted_box):
            tracker_bbox = predicted_box
            box_iou = float(np.nan_to_num(bbox_compute_iou(predicted_box, bbox), nan=0.0))

    return _TrackerCandidateSupport(response_score, relative_response, box_iou, tracker_bbox)
