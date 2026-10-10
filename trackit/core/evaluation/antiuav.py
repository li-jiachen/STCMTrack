
from dataclasses import dataclass
import numpy as np

PROTOCOL = {
    'id': 'stcmtrack-antiuav',
    'boxes': 'continuous_xywh',
    'auc': 'exact_integral_of_empirical_success_curve_equals_mean_iou',
    'success_plot': '101 thresholds in [0,1], IoU >= threshold',
    'precision': 'center_error < 20 pixels',
    'normalized_precision': 'center_error / hypot(gt_width, gt_height) < 0.5',
    'absent_frames': 'excluded from all metric denominators',
    'initialization_frame': 'included',
    'aggregation': 'unweighted mean over sequences',
    'coverage': 'complete sequence set and equal frame lengths unless explicitly partial',
}
IOU_THRESHOLDS = np.linspace(0., 1., 101)
CENTER_THRESHOLDS = np.arange(51, dtype=np.float64)
NORM_THRESHOLDS = np.linspace(0., .5, 51)


@dataclass(frozen=True)
class SequenceMetrics:
    iou: np.ndarray
    center_error: np.ndarray
    normalized_error: np.ndarray
    valid: np.ndarray
    success_curve: np.ndarray
    precision_curve: np.ndarray
    normalized_precision_curve: np.ndarray
    auc: float
    p20: float
    pn: float


def frame_errors(pred, gt, exists=None):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape or pred.shape[1:] != (4,) or not len(gt):
        raise ValueError(f'Predictions and GT must have identical nonempty [frames,4] shapes: {pred.shape}, {gt.shape}')
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite prediction coordinates')
    finite_gt = np.isfinite(gt).all(axis=1)
    if exists is None and not finite_gt.all():
        raise ValueError('Non-finite GT requires an explicit absence flag')
    valid = finite_gt & (gt[:, 2:] > 0).all(axis=1)
    if exists is not None:
        exists = np.asarray(exists)
        if exists.shape != (len(gt),) or not np.isin(exists, (0, 1)).all():
            raise ValueError('exist must have exactly one binary flag per frame')
        if np.any(exists.astype(bool) & ~valid):
            raise ValueError('A present target must have positive ground-truth width and height')
        valid &= exists.astype(bool)
    pred_valid = (pred[:, 2:] > 0).all(axis=1)
    active = valid & pred_valid
    iou = np.zeros(len(gt), dtype=np.float64)
    center = np.full(len(gt), np.inf)
    normalized = np.full(len(gt), np.inf)
    if active.any():
        p, g = pred[active], gt[active]
        tl = np.maximum(p[:, :2], g[:, :2])
        br = np.minimum(p[:, :2] + p[:, 2:], g[:, :2] + g[:, 2:])
        intersection = np.maximum(br - tl, 0.).prod(axis=1)
        union = p[:, 2:].prod(axis=1) + g[:, 2:].prod(axis=1) - intersection
        iou[active] = np.clip(intersection / union, 0., 1.)
        center[active] = np.linalg.norm((p[:, :2] + p[:, 2:] / 2) - (g[:, :2] + g[:, 2:] / 2), axis=1)
        normalized[active] = center[active] / np.hypot(g[:, 2], g[:, 3])
    return iou, center, normalized, valid


def evaluate_sequence(pred, gt, exists=None):
    iou, center, normalized, valid = frame_errors(pred, gt, exists)
    if not valid.any():
        raise ValueError('Cannot evaluate a sequence with no present, valid ground-truth frames')
    vi, vc, vn = iou[valid], center[valid], normalized[valid]
    return SequenceMetrics(
        iou, center, normalized, valid,
        (vi[:, None] >= IOU_THRESHOLDS).mean(axis=0),
        (vc[:, None] < CENTER_THRESHOLDS).mean(axis=0),
        (vn[:, None] < NORM_THRESHOLDS).mean(axis=0),
        float(vi.mean()), float((vc < 20.).mean()), float((vn < .5).mean()))
