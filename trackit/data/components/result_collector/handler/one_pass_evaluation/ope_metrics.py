from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import numpy as np
from trackit.core.operator.numpy.bbox.iou import bbox_compute_iou
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.miscellanies.numpy_array_builder import NumpyArrayBuilder
from trackit.miscellanies.argsort import argsort


from trackit.core.evaluation.antiuav import evaluate_sequence, frame_errors
from trackit.core.operator.numpy.bbox.format import bbox_xyxy_to_xywh

_bins_of_center_location_error = 51
_bins_of_normalized_center_location_error = 51
_bins_of_intersection_of_union = 101
_bin_index_precision_score = 20
_bin_index_normalized_precision_score = 50


def calculate_center_location_error(pred_bb, anno_bb, normalized=False):
    _, center, norm, _ = frame_errors(bbox_xyxy_to_xywh(pred_bb), bbox_xyxy_to_xywh(anno_bb))
    return norm if normalized else center


def compute_one_pass_evaluation_metrics(predicted_bounding_boxes, groundtruth_bounding_boxes,
                                        bounding_box_validity_flags, time_costs):
    result = evaluate_sequence(bbox_xyxy_to_xywh(predicted_bounding_boxes),
                               bbox_xyxy_to_xywh(groundtruth_bounding_boxes), bounding_box_validity_flags)
    overlap = result.iou.copy()
    overlap[~result.valid] = -1.
    return OPEMetrics(result.auc, float(result.success_curve[50]), float(result.success_curve[75]),
                      result.success_curve, result.precision_curve, result.normalized_precision_curve,
                      float(np.mean(time_costs))), overlap


@dataclass(frozen=True)
class OPEMetrics:
    average_overlap: float
    success_rate_at_iou_0_5: float
    success_rate_at_iou_0_75: float
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: float

    @property
    def success_score(self) -> float:
        return float(self.average_overlap)

    @property
    def precision_score(self) -> float:
        return self.precision_curve[_bin_index_precision_score].item()

    @property
    def normalized_precision_score(self) -> float:
        return self.normalized_precision_curve[_bin_index_normalized_precision_score].item()

    def get_fps(self) -> float:
        return 1. / self.time_cost if self.time_cost > 0 else 0.


def compute_OPE_metrics_mean(metrics: Sequence[OPEMetrics]) -> OPEMetrics:
    return OPEMetrics(
        average_overlap=np.mean([m.average_overlap for m in metrics]).item(),
        success_rate_at_iou_0_5=np.mean([m.success_rate_at_iou_0_5 for m in metrics]).item(),
        success_rate_at_iou_0_75=np.mean([m.success_rate_at_iou_0_75 for m in metrics]).item(),
        success_curve=np.mean([m.success_curve for m in metrics], axis=0),
        precision_curve=np.mean([m.precision_curve for m in metrics], axis=0),
        normalized_precision_curve=np.mean([m.normalized_precision_curve for m in metrics], axis=0),
        time_cost=np.mean([m.time_cost for m in metrics]).item()
    )


@dataclass(frozen=True)
class DatasetOPEMetricsList:
    sequence_name: Sequence[str]
    average_overlap: np.ndarray
    success_rate_at_iou_0_5: np.ndarray
    success_rate_at_iou_0_75: np.ndarray
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    time_cost: np.ndarray

    def __getitem__(self, index: int):
        return self.sequence_name[index], \
            OPEMetrics(self.average_overlap[index].item(), self.success_rate_at_iou_0_5[index].item(),
                            self.success_rate_at_iou_0_75[index].item(), self.success_curve[index], self.precision_curve[index],
                            self.normalized_precision_curve[index], self.time_cost[index].item())

    def __len__(self):
        return len(self.sequence_name)

    def sort_by_sequence_name(self):
        sorted_indices = np.array(argsort(self.sequence_name))

        return DatasetOPEMetricsList(
            tuple(self.sequence_name[index] for index in sorted_indices),
            self.average_overlap[sorted_indices],
            self.success_rate_at_iou_0_5[sorted_indices],
            self.success_rate_at_iou_0_75[sorted_indices],
            self.success_curve[sorted_indices],
            self.precision_curve[sorted_indices],
            self.normalized_precision_curve[sorted_indices],
            self.time_cost[sorted_indices])

    def get_mean(self):
        return OPEMetrics(np.mean(self.average_overlap).item(), np.mean(self.success_rate_at_iou_0_5).item(),
                            np.mean(self.success_rate_at_iou_0_75).item(), np.mean(self.success_curve, axis=0),
                            np.mean(self.precision_curve, axis=0), np.mean(self.normalized_precision_curve, axis=0),
                            np.mean(self.time_cost).item())

    def get_success_score(self) -> np.ndarray:
        return self.average_overlap

    def get_precision_score(self) -> np.ndarray:
        return self.precision_curve[:, _bin_index_precision_score]

    def get_normalized_precision_score(self) -> np.ndarray:
        return self.normalized_precision_curve[:, _bin_index_normalized_precision_score]


class DatasetOPEMetricsListBuilder:
    def __init__(self):
        self._sequence_name = []
        self._average_overlap = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_5 = NumpyArrayBuilder(np.float64)
        self._success_rate_at_iou_0_75 = NumpyArrayBuilder(np.float64)
        self._success_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_intersection_of_union,))
        self._precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_center_location_error,))
        self._norm_precision_curve = NumpyArrayBuilder(np.float64, extra_dims=(_bins_of_normalized_center_location_error,))
        self._average_time_cost = NumpyArrayBuilder(np.float64)

    def append(self, sequence_name: str, metrics: OPEMetrics):
        self._sequence_name.append(sequence_name)
        self._average_overlap.append(metrics.average_overlap)
        self._success_rate_at_iou_0_5.append(metrics.success_rate_at_iou_0_5)
        self._success_rate_at_iou_0_75.append(metrics.success_rate_at_iou_0_75)
        self._success_curve.append(metrics.success_curve)
        self._precision_curve.append(metrics.precision_curve)
        self._norm_precision_curve.append(metrics.normalized_precision_curve)
        self._average_time_cost.append(metrics.time_cost)

    def build(self):
        return DatasetOPEMetricsList(self._sequence_name, self._average_overlap.build(),
                                     self._success_rate_at_iou_0_5.build(),
                                     self._success_rate_at_iou_0_75.build(), self._success_curve.build(),
                                     self._precision_curve.build(), self._norm_precision_curve.build(),
                                     self._average_time_cost.build())
