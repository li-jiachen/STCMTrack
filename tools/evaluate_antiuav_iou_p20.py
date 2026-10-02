#!/usr/bin/env python3
"""Strict Anti-UAV evaluation with the shared STCMTrack paper-v1 protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackit.core.evaluation.antiuav import evaluate_sequence as paper_evaluate_sequence, frame_errors, PROTOCOL

DEFAULT_GT_DIR = Path(__file__).resolve().parents[2] / "antiuav410" / "test"



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


def compute_iou(rect1, rect2):
    return float(frame_errors([rect1], [rect2])[0][0])


def compute_center_error(rect1, rect2):
    return float(frame_errors([rect1], [rect2])[1][0])


def compute_norm_center_error(rect1, rect2):
    return float(frame_errors([rect1], [rect2])[2][0])


def compute_frame_metrics(pred, gt):
    errors = frame_errors([pred], [gt])
    return tuple(float(value[0]) for value in errors[:3])


def evaluate_sequence(seq_name, pred_name, pred_boxes, gt_boxes, exists=None, *, allow_partial=False):
    pred_length, gt_length = len(pred_boxes), len(gt_boxes)
    if exists is not None and len(exists) != gt_length:
        raise ValueError(f'{seq_name}: exist/gt_rect length mismatch')
    if pred_length != gt_length:
        if not allow_partial or not 0 < pred_length <= gt_length:
            raise ValueError(f'{seq_name}: prediction/GT length mismatch {pred_length}/{gt_length}')
    used = pred_length
    result = paper_evaluate_sequence(pred_boxes, gt_boxes[:used], None if exists is None else exists[:used])
    return AntiUAVSequenceMetric(seq_name, pred_name, result.auc, result.p20, result.pn,
                                 int(result.valid.sum()), used, pred_length, gt_length)


def parse_prediction_text(text):
    boxes = []
    for index, line in enumerate(text.splitlines(), 1):
        parts = line.replace(',', ' ').split()
        if len(parts) != 4:
            raise ValueError(f'Prediction line {index} must contain exactly four coordinates')
        try:
            box = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f'Invalid prediction number on line {index}') from exc
        if not np.isfinite(box).all():
            raise ValueError(f'Non-finite prediction on line {index}')
        boxes.append(box)
    if not boxes:
        raise ValueError('Empty prediction file')
    return boxes, 0


def aggregate_metrics(
    sequence_metrics: list[AntiUAVSequenceMetric],
    skipped: int = 0,
    invalid_prediction_lines: int = 0,
) -> AntiUAVMetric:
    if not sequence_metrics:
        raise ValueError('No sequence metrics to aggregate')

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


def evaluate_antiuav_results(pred_zip: Path, gt_dir: Path, *, allow_partial=False, expected_sequences=None):
    gt_dirs = [path for path in gt_dir.iterdir() if path.is_dir()]
    missing_labels = [path.name for path in gt_dirs if not (path / 'IR_label.json').is_file()]
    if missing_labels:
        raise ValueError(f'GT sequence directories without IR_label.json: {sorted(missing_labels)}')
    gt_folders = {path.name for path in gt_dirs}
    if not gt_folders:
        raise ValueError(f'No IR_label.json sequence directories in {gt_dir}')
    if expected_sequences is not None and len(gt_folders) != expected_sequences:
        raise ValueError(f'GT split contains {len(gt_folders)} sequences; expected {expected_sequences}')
    metrics, seen = [], set()
    with zipfile.ZipFile(pred_zip) as archive:
        for pred_path in sorted(name for name in archive.namelist() if name.endswith('.txt')):
            name = Path(pred_path).stem
            if name not in gt_folders:
                raise ValueError(f'Unknown prediction sequence {name!r}; exact names are required')
            if name in seen:
                raise ValueError(f'Duplicate prediction for sequence {name!r}')
            seen.add(name)
            boxes, _ = parse_prediction_text(archive.read(pred_path).decode('utf-8'))
            with (gt_dir / name / 'IR_label.json').open(encoding='utf-8') as handle:
                labels = json.load(handle)
            if 'exist' not in labels or 'gt_rect' not in labels:
                raise ValueError(f'{name}: labels must contain exist and gt_rect')
            metrics.append(evaluate_sequence(name, name, boxes, labels['gt_rect'], labels['exist'],
                                             allow_partial=allow_partial))
    if not allow_partial and seen != gt_folders:
        raise ValueError(f'Incomplete evaluation; missing sequences: {sorted(gt_folders - seen)}')
    return aggregate_metrics(metrics, skipped=len(gt_folders - seen)), metrics


def write_manifest(path, pred_zip, gt_dir, metric, sequence_metrics, allow_partial):
    gt_hashes = {}
    for row in sequence_metrics:
        label = gt_dir / row.seq_name / 'IR_label.json'
        gt_hashes[row.seq_name] = hashlib.sha256(label.read_bytes()).hexdigest()
    report = {'protocol': PROTOCOL, 'partial_evaluation': allow_partial,
              'predictions_sha256': hashlib.sha256(pred_zip.read_bytes()).hexdigest(),
              'groundtruth_sha256_by_sequence': gt_hashes,
              'summary': asdict(metric), 'sequences': [asdict(row) for row in sequence_metrics]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


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


def _run_self_tests():
    same = evaluate_sequence('s', 's', [[0, 0, 10, 10]], [[0, 0, 10, 10]])
    miss = evaluate_sequence('s', 's', [[100, 100, 10, 10]], [[0, 0, 10, 10]])
    assert (same.auc, same.precision_at_20, same.norm_precision_at_05) == (1., 1., 1.)
    assert (miss.auc, miss.precision_at_20, miss.norm_precision_at_05) == (0., 0., 0.)
    # A 30x40 GT has diagonal 50. Error 25 sits exactly on Pn's excluded boundary.
    edge = evaluate_sequence('s', 's', [[25, 0, 30, 40]], [[0, 0, 30, 40]])
    assert edge.norm_precision_at_05 == 0.
    assert evaluate_sequence('s', 's', [[20, 0, 30, 40]], [[0, 0, 30, 40]]).precision_at_20 == 0.
    visible = evaluate_sequence('s', 's', [[0, 0, 10, 10], [100, 100, 10, 10]],
                                [[0, 0, 10, 10], [0, 0, 10, 10]], [1, 0])
    assert visible.valid_frames == 1 and visible.auc == 1.
    print('All metric self-tests passed.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_zip", nargs="?", type=Path)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(os.environ.get("ANTIUAV_GT_DIR", str(DEFAULT_GT_DIR))),
    )
    parser.add_argument("--sequence-csv", type=Path)
    parser.add_argument('--report-json', type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Explicit smoke-test mode; never report as a full benchmark')
    parser.add_argument('--expected-sequences', type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return

    if args.pred_zip is None:
        parser.error("pred_zip is required unless --self-test is set")

    metric, sequence_metrics = evaluate_antiuav_results(args.pred_zip, args.gt_dir, allow_partial=args.allow_partial,
                                                               expected_sequences=args.expected_sequences)

    if args.sequence_csv is not None:
        write_sequence_csv(args.sequence_csv, metric, sequence_metrics)

    report_path = args.report_json or args.pred_zip.with_suffix('.paper_metrics.json')
    write_manifest(report_path, args.pred_zip, args.gt_dir, metric, sequence_metrics, args.allow_partial)
    if args.allow_partial:
        print('PARTIAL / SMOKE-TEST RESULTS: not a full benchmark')
    print_summary(metric, sequence_metrics)
    print(f'Protocol and coverage manifest: {report_path}')


if __name__ == "__main__":
    main()
