#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUFFIXES = ["_train_newfix", "_test", "_train", "_val", "_train-new", "_newfix", "_new"]


@dataclass
class AntiUAVMetric:
    auc: float
    precision_at_20: float
    norm_precision_at_05: float
    matched_sequences: int
    tracking_frames: int
    valid_frames: int
    skipped: int
    invalid_prediction_lines: int


@dataclass
class AntiUAVSequenceMetric:
    seq_name: str
    pred_name: str
    auc: float
    precision_at_20: float
    norm_precision_at_05: float
    valid_frames: int
    used_frames: int
    pred_frames: int
    gt_frames: int


def compute_iou(rect1: list[float], rect2: list[float]) -> float:
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_center_error(rect1: list[float], rect2: list[float]) -> float:
    c1_x = rect1[0] + rect1[2] / 2
    c1_y = rect1[1] + rect1[3] / 2
    c2_x = rect2[0] + rect2[2] / 2
    c2_y = rect2[1] + rect2[3] / 2
    return float(np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2))


def compute_norm_center_error(rect1: list[float], rect2: list[float]) -> float:
    w, h = rect2[2], rect2[3]
    if w <= 0 or h <= 0:
        return float("inf")
    return compute_center_error(rect1, rect2) / math.sqrt(w * h)


def calculate_auc(success_rates: np.ndarray, thresholds: np.ndarray) -> float:
    return float(np.trapz(success_rates, thresholds) / (thresholds[-1] - thresholds[0]))


def compute_frame_metrics(pred: list[float], gt: list[float]) -> tuple[float, float, float]:
    if pred[2] <= 0 or pred[3] <= 0:
        return 0.0, float("inf"), float("inf")
    iou = compute_iou(pred, gt)
    center_error = compute_center_error(pred, gt)
    norm_center_error = compute_norm_center_error(pred, gt)
    return iou, center_error, norm_center_error


def evaluate_sequence(
    seq_name: str, pred_name: str, pred_boxes: list[list[float]], gt_boxes: list[list[float]]
) -> AntiUAVSequenceMetric | None:
    used_frames = min(len(pred_boxes), len(gt_boxes))
    ious: list[float] = []
    center_errors: list[float] = []
    norm_center_errors: list[float] = []

    for pred, gt in zip(pred_boxes[:used_frames], gt_boxes[:used_frames]):
        if gt[2] <= 0 or gt[3] <= 0:
            continue
        iou, center_error, norm_center_error = compute_frame_metrics(pred, gt)
        ious.append(iou)
        center_errors.append(center_error)
        norm_center_errors.append(norm_center_error)

    if not ious:
        return None

    iou_thresholds = np.linspace(0.0, 1.0, 101)
    iou_array = np.asarray(ious, dtype=float)
    center_error_array = np.asarray(center_errors, dtype=float)
    norm_center_error_array = np.asarray(norm_center_errors, dtype=float)
    success_rates = np.asarray(
        [np.sum(iou_array >= threshold) / len(iou_array) for threshold in iou_thresholds],
        dtype=float,
    )

    return AntiUAVSequenceMetric(
        seq_name=seq_name,
        pred_name=pred_name,
        auc=calculate_auc(success_rates, iou_thresholds),
        precision_at_20=float(np.sum(center_error_array <= 20.0) / len(center_error_array)),
        norm_precision_at_05=float(np.sum(norm_center_error_array <= 0.5) / len(norm_center_error_array)),
        valid_frames=len(ious),
        used_frames=used_frames,
        pred_frames=len(pred_boxes),
        gt_frames=len(gt_boxes),
    )


def auto_match_folder(pred_name: str, gt_folders: set[str]) -> str | None:
    candidates = [pred_name]
    if pred_name.startswith("uav_"):
        candidates.append(pred_name[4:])

    for candidate in list(candidates):
        for suffix in SUFFIXES:
            if candidate.endswith(suffix):
                base = candidate[: -len(suffix)]
                if base not in candidates:
                    candidates.append(base)
                if base.startswith("uav_") and base[4:] not in candidates:
                    candidates.append(base[4:])

    for candidate in set(candidates):
        if candidate in gt_folders:
            return candidate
    return None


def parse_prediction_text(text: str) -> tuple[list[list[float]], int]:
    boxes: list[list[float]] = []
    invalid_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            invalid_lines += 1
            continue
        boxes.append([float(part) for part in parts])
    return boxes, invalid_lines


def aggregate_metrics(
    sequence_metrics: list[AntiUAVSequenceMetric],
    skipped: int = 0,
    invalid_prediction_lines: int = 0,
) -> AntiUAVMetric:
    if not sequence_metrics:
        return AntiUAVMetric(
            auc=math.nan,
            precision_at_20=math.nan,
            norm_precision_at_05=math.nan,
            matched_sequences=0,
            tracking_frames=0,
            valid_frames=0,
            skipped=skipped,
            invalid_prediction_lines=invalid_prediction_lines,
        )

    return AntiUAVMetric(
        auc=float(np.mean([result.auc for result in sequence_metrics])),
        precision_at_20=float(np.mean([result.precision_at_20 for result in sequence_metrics])),
        norm_precision_at_05=float(
            np.mean([result.norm_precision_at_05 for result in sequence_metrics])
        ),
        matched_sequences=len(sequence_metrics),
        tracking_frames=sum(result.used_frames for result in sequence_metrics),
        valid_frames=sum(result.valid_frames for result in sequence_metrics),
        skipped=skipped,
        invalid_prediction_lines=invalid_prediction_lines,
    )


def evaluate_antiuav_results(
    pred_zip: Path, gt_dir: Path
) -> tuple[AntiUAVMetric, list[AntiUAVSequenceMetric]]:
    gt_folders = {path.name for path in gt_dir.iterdir() if path.is_dir()}
    sequence_metrics: list[AntiUAVSequenceMetric] = []
    skipped = 0
    invalid_prediction_lines = 0

    with zipfile.ZipFile(pred_zip) as zip_file:
        pred_names = sorted(
            name for name in zip_file.namelist() if name.endswith(".txt") and not name.endswith("/")
        )
        for pred_path in pred_names:
            pred_name = Path(pred_path).stem
            seq_name = auto_match_folder(pred_name, gt_folders)
            if seq_name is None:
                skipped += 1
                continue

            gt_path = gt_dir / seq_name / "IR_label.json"
            if not gt_path.exists():
                skipped += 1
                continue

            pred_boxes, invalid_lines = parse_prediction_text(
                zip_file.read(pred_path).decode("utf-8", errors="replace")
            )
            invalid_prediction_lines += invalid_lines
            with gt_path.open("r", encoding="utf-8") as file:
                gt_boxes = json.load(file)["gt_rect"]

            result = evaluate_sequence(seq_name, pred_name, pred_boxes, gt_boxes)
            if result is None:
                skipped += 1
            else:
                sequence_metrics.append(result)

    metric = aggregate_metrics(
        sequence_metrics,
        skipped=skipped,
        invalid_prediction_lines=invalid_prediction_lines,
    )
    return metric, sequence_metrics


def write_sequence_csv(
    path: Path, metric: AntiUAVMetric, sequence_metrics: list[AntiUAVSequenceMetric]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sequence",
                "valid_frames",
                "auc",
                "precision_at_20",
                "normalized_precision_at_0.5",
                "pred_name",
                "used_frames",
                "pred_frames",
                "gt_frames",
            ]
        )
        for result in sorted(sequence_metrics, key=lambda item: item.auc, reverse=True):
            writer.writerow(
                [
                    result.seq_name,
                    result.valid_frames,
                    f"{result.auc:.6f}",
                    f"{result.precision_at_20:.6f}",
                    f"{result.norm_precision_at_05:.6f}",
                    result.pred_name,
                    result.used_frames,
                    result.pred_frames,
                    result.gt_frames,
                ]
            )
        writer.writerow(
            [
                "__overall__",
                metric.valid_frames,
                f"{metric.auc:.6f}",
                f"{metric.precision_at_20:.6f}",
                f"{metric.norm_precision_at_05:.6f}",
                "",
                metric.tracking_frames,
                "",
                "",
            ]
        )


def print_summary(metric: AntiUAVMetric, sequence_metrics: list[AntiUAVSequenceMetric]) -> None:
    print("Aggregation mode: sequence-level macro average")
    for result in sorted(sequence_metrics, key=lambda item: item.auc, reverse=True):
        print(
            f"sequence={result.seq_name} "
            f"valid_frames={result.valid_frames} "
            f"AUC={result.auc:.6f} "
            f"P@20={result.precision_at_20:.6f} "
            f"NP@0.5={result.norm_precision_at_05:.6f}"
        )

    print(f"matched_sequences={metric.matched_sequences}")
    print(f"skipped_sequences={metric.skipped}")
    print(f"tracking_frames={metric.tracking_frames}")
    print(f"valid_frames={metric.valid_frames}")
    print(f"AUC={metric.auc:.6f}")
    print(f"P@20={metric.precision_at_20:.6f}")
    print(f"NP@0.5={metric.norm_precision_at_05:.6f}")
    print(f"invalid_prediction_lines={metric.invalid_prediction_lines}")


def _run_self_tests() -> None:
    identical = evaluate_sequence("seq_a", "pred_a", [[0, 0, 10, 10]], [[0, 0, 10, 10]])
    assert identical is not None
    assert math.isclose(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    assert math.isclose(compute_center_error([0, 0, 10, 10], [0, 0, 10, 10]), 0.0)
    assert math.isclose(compute_norm_center_error([0, 0, 10, 10], [0, 0, 10, 10]), 0.0)
    assert math.isclose(identical.auc, 1.0)
    assert math.isclose(identical.precision_at_20, 1.0)
    assert math.isclose(identical.norm_precision_at_05, 1.0)

    disjoint = evaluate_sequence("seq_b", "pred_b", [[100, 100, 10, 10]], [[0, 0, 10, 10]])
    assert disjoint is not None
    assert math.isclose(compute_iou([100, 100, 10, 10], [0, 0, 10, 10]), 0.0)
    assert math.isclose(disjoint.precision_at_20, 0.0)
    assert math.isclose(disjoint.norm_precision_at_05, 0.0)
    assert math.isclose(disjoint.auc, 0.005)

    invalid_pred_iou, invalid_pred_center_error, invalid_pred_norm_error = compute_frame_metrics([0, 0, 0, 0], [0, 0, 10, 10])
    assert math.isclose(invalid_pred_iou, 0.0)
    assert math.isinf(invalid_pred_center_error)
    assert math.isinf(invalid_pred_norm_error)
    invalid_pred = evaluate_sequence("seq_c", "pred_c", [[0, 0, 0, 0]], [[0, 0, 10, 10]])
    assert invalid_pred is not None
    assert math.isclose(invalid_pred.precision_at_20, 0.0)
    assert math.isclose(invalid_pred.norm_precision_at_05, 0.0)
    assert math.isclose(invalid_pred.auc, 0.005)

    invalid_gt = evaluate_sequence("seq_d", "pred_d", [[0, 0, 10, 10]], [[0, 0, 0, 10]])
    assert invalid_gt is None

    seq_hi = AntiUAVSequenceMetric("seq_hi", "pred_hi", 1.0, 1.0, 1.0, 1, 1, 1, 1)
    seq_lo = AntiUAVSequenceMetric("seq_lo", "pred_lo", 0.0, 0.0, 0.0, 1000, 1000, 1000, 1000)
    macro = aggregate_metrics([seq_hi, seq_lo])
    assert math.isclose(macro.auc, 0.5)
    assert math.isclose(macro.precision_at_20, 0.5)
    assert math.isclose(macro.norm_precision_at_05, 0.5)

    print("All self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_zip", nargs="?", type=Path)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(os.environ.get("ANTIUAV_GT_DIR", "/root/lanyun-fs/antiuav410/test")),
    )
    parser.add_argument("--sequence-csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return

    if args.pred_zip is None:
        parser.error("pred_zip is required unless --self-test is set")

    metric, sequence_metrics = evaluate_antiuav_results(args.pred_zip, args.gt_dir)

    if args.sequence_csv is not None:
        write_sequence_csv(args.sequence_csv, metric, sequence_metrics)

    print_summary(metric, sequence_metrics)


if __name__ == "__main__":
    main()
