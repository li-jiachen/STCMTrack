# modify from https://github.com/visionml/pytracking/blob/master/pytracking/analysis/extract_results.py
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import numpy as np
from trackit.miscellanies.numpy_array_builder import NumpyArrayBuilder
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh
from trackit.miscellanies.argsort import argsort
from ..utils.compatibility import ExternalToolkitCompatibilityHelper


from trackit.core.evaluation.antiuav import (
    evaluate_sequence, frame_errors, IOU_THRESHOLDS, CENTER_THRESHOLDS, NORM_THRESHOLDS)

_plot_bin_gap = 0.01
_bins_of_intersection_of_union = len(IOU_THRESHOLDS)
_bins_of_center_location_error = len(CENTER_THRESHOLDS)
_threshold_set_overlap = IOU_THRESHOLDS
_threshold_set_center = CENTER_THRESHOLDS
_threshold_set_center_norm = NORM_THRESHOLDS
_precision_score_bin_index = 20
_normalized_precision_score_bin_index = 50
_success_rate_at_overlap_0_5_bin_index = 50
_success_rate_at_overlap_0_75_bin_index = 75


def calc_err_center(pred_bb, anno_bb, normalized=False):
    _, center, norm, _ = frame_errors(pred_bb, anno_bb)
    return norm if normalized else center


def calc_iou_overlap(pred_bb, anno_bb):
    return frame_errors(pred_bb, anno_bb)[0]


def calc_seq_err_robust(dataset_name, pred_bb, anno_bb, target_visible=None, stark_behavior=True):
    return frame_errors(pred_bb, anno_bb, target_visible)


def compute_one_pass_evaluation_metrics(dataset_name, pred_bb, anno_bb, target_visible,
                                        time_costs, compatibility_helper, stark_behavior=True,
                                        exclude_invalid_frames=True):
    
    pred_bb = bbox_xyxy_to_xywh(compatibility_helper.adjust(dataset_name, pred_bb))
    anno_bb = bbox_xyxy_to_xywh(compatibility_helper.adjust(dataset_name, anno_bb))
    result = evaluate_sequence(pred_bb, anno_bb, target_visible)
    overlap = result.iou.copy()
    overlap[~result.valid] = -1.  
    return OPEMetrics(result.success_curve, result.precision_curve, result.normalized_precision_curve,
                      float(np.mean(time_costs)), result.auc), overlap


@dataclass(frozen=True)
class OPEMetrics:
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: float
    auc: float

    @property
    def success_score(self) -> float:
        return float(self.auc)

    @property
    def precision_score(self) -> float:
        return self.precision_curve[20].item()

    @property
    def normalized_precision_score(self) -> float:
        return self.normalized_precision_curve[_normalized_precision_score_bin_index].item()

    @property
    def success_rate_at_overlap_0_5(self) -> float:
        return self.success_curve[_success_rate_at_overlap_0_5_bin_index].item()

    @property
    def success_rate_at_overlap_0_75(self) -> float:
        return self.success_curve[_success_rate_at_overlap_0_75_bin_index].item()

    def get_fps(self) -> float:
        return 1. / self.time_cost if self.time_cost > 0 else 0.


def compute_OPE_metrics_mean(metrics: Sequence[OPEMetrics]) -> OPEMetrics:
    return OPEMetrics(
        success_curve=np.mean([m.success_curve for m in metrics], axis=0),
        precision_curve=np.mean([m.precision_curve for m in metrics], axis=0),
        normalized_precision_curve=np.mean([m.normalized_precision_curve for m in metrics], axis=0),
        time_cost=np.mean([m.time_cost for m in metrics]).item(),
        auc=np.mean([m.auc for m in metrics]).item()
    )


@dataclass(frozen=True)
class DatasetOPEMetricsList:
    sequence_name: Sequence[str]
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: np.ndarray
    auc: np.ndarray

    def __getitem__(self, index: int):
        return self.sequence_name[index], \
            OPEMetrics(self.success_curve[index], self.precision_curve[index], self.normalized_precision_curve[index], self.time_cost[index].item(), self.auc[index].item())

    def __len__(self):
        return len(self.sequence_name)

    def sort_by_sequence_name(self):
        sorted_indices = np.array(argsort(self.sequence_name))

        return DatasetOPEMetricsList(
            tuple(self.sequence_name[index] for index in sorted_indices),
            self.success_curve[sorted_indices],
            self.precision_curve[sorted_indices],
            self.normalized_precision_curve[sorted_indices],
            self.time_cost[sorted_indices], self.auc[sorted_indices])

    def get_mean(self):
        return OPEMetrics(np.mean(self.success_curve, axis=0),
                          np.mean(self.precision_curve, axis=0),
                          np.mean(self.normalized_precision_curve, axis=0),
                          np.mean(self.time_cost).item(), np.mean(self.auc).item())

    def get_success_score(self) -> np.ndarray:
        return self.auc

    def get_precision_score(self) -> np.ndarray:
        return self.precision_curve[:, _precision_score_bin_index]

    def get_normalized_precision_score(self) -> np.ndarray:
        return self.normalized_precision_curve[:, _normalized_precision_score_bin_index]


class DatasetOPEMetricsListBuilder:
    def __init__(self):
        self._sequence_name = []
        self._average_overlap = NumpyArrayBuilder(np.float64)
        self._success_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_intersection_of_union,))
        self._precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._norm_precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._average_time_cost = NumpyArrayBuilder(np.float64)

    def append(self, sequence_name: str, metrics: OPEMetrics):
        self._sequence_name.append(sequence_name)
        self._average_overlap.append(metrics.auc)
        self._success_curve.append(metrics.success_curve)
        self._precision_curve.append(metrics.precision_curve)
        self._norm_precision_curve.append(metrics.normalized_precision_curve)
        self._average_time_cost.append(metrics.time_cost)

    def build(self):
        return DatasetOPEMetricsList(self._sequence_name, self._success_curve.build(),
                                     self._precision_curve.build(), self._norm_precision_curve.build(),
                                     self._average_time_cost.build(), self._average_overlap.build())
