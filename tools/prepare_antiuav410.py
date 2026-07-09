#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_SPLITS = ('train', 'val', 'test')


@dataclass
class SequenceReport:
    split: str
    name: str
    source: str
    target: str
    frame_count: int
    annotation_count: int
    valid_annotation_count: int
    missing_images: int


@dataclass
class SplitReport:
    split: str
    source: str
    target: str
    sequence_count: int
    frame_count: int
    annotation_count: int
    valid_annotation_count: int
    missing_images: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare an Anti-UAV 410 dataset tree for STCMTrack.'
    )
    parser.add_argument('--source-root', required=True,
                        help='Path to the raw or already organized Anti-UAV 410 dataset root.')
    parser.add_argument('--output-root', required=True,
                        help='Path where the standardized dataset tree will be created.')
    parser.add_argument('--mode', choices=('symlink', 'copy'), default='symlink',
                        help='How to materialize sequences under the output root.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing output tree.')
    parser.add_argument('--single-split-name', default='test',
                        help='Split name to use when the source root does not contain train/val/test subdirectories.')
    parser.add_argument('--write-manifest', action='store_true',
                        help='Write a JSON manifest next to the prepared dataset tree.')
    return parser.parse_args()


def _read_annotation(annotation_path: Path) -> tuple[list[list[float]], list[int]]:
    with annotation_path.open('r', encoding='utf-8') as file:
        data = json.load(file)

    boxes = data.get('gt_rect', [])
    exists = data.get('exist', [])

    if len(boxes) != len(exists):
        raise ValueError(
            f'Annotation length mismatch in {annotation_path}: gt_rect={len(boxes)} exist={len(exists)}'
        )

    normalized_boxes: list[list[float]] = []
    normalized_exists: list[int] = []
    for box, exist in zip(boxes, exists):
        if len(box) != 4:
            raise ValueError(f'Invalid bbox in {annotation_path}: {box}')
        normalized_boxes.append([float(value) for value in box])
        normalized_exists.append(int(exist))
    return normalized_boxes, normalized_exists


def _iter_sequence_dirs(split_root: Path) -> list[Path]:
    return sorted(
        [path for path in split_root.iterdir() if path.is_dir() and not path.name.startswith('.')],
        key=lambda path: path.name,
    )


def _iter_images(sequence_dir: Path) -> list[Path]:
    images = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda path: path.name)


def _ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f'Output path already exists: {path}')
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_sequence(source_dir: Path, target_dir: Path, mode: str) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        os.symlink(source_dir.resolve(), target_dir, target_is_directory=True)
    else:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _prepare_split(split: str, source_root: Path, output_root: Path, mode: str) -> tuple[SplitReport, list[SequenceReport]]:
    split_source = source_root / split if (source_root / split).is_dir() else source_root
    split_target = output_root / split
    split_target.mkdir(parents=True, exist_ok=True)

    sequence_reports: list[SequenceReport] = []
    for sequence_dir in _iter_sequence_dirs(split_source):
        annotation_path = sequence_dir / 'IR_label.json'
        if not annotation_path.exists():
            raise FileNotFoundError(f'Missing annotation file: {annotation_path}')

        boxes, exists = _read_annotation(annotation_path)
        images = _iter_images(sequence_dir)
        image_name_set = {path.name for path in images}
        missing_images = 0
        for index in range(len(boxes)):
            image_name = f'{index + 1:06d}.jpg'
            if image_name not in image_name_set:
                missing_images += 1

        target_sequence_dir = split_target / sequence_dir.name
        _materialize_sequence(sequence_dir, target_sequence_dir, mode)

        sequence_reports.append(
            SequenceReport(
                split=split,
                name=sequence_dir.name,
                source=str(sequence_dir),
                target=str(target_sequence_dir),
                frame_count=len(boxes),
                annotation_count=len(boxes),
                valid_annotation_count=sum(1 for value in exists if value),
                missing_images=missing_images,
            )
        )

    split_report = SplitReport(
        split=split,
        source=str(split_source),
        target=str(split_target),
        sequence_count=len(sequence_reports),
        frame_count=sum(report.frame_count for report in sequence_reports),
        annotation_count=sum(report.annotation_count for report in sequence_reports),
        valid_annotation_count=sum(report.valid_annotation_count for report in sequence_reports),
        missing_images=sum(report.missing_images for report in sequence_reports),
    )
    return split_report, sequence_reports


def _discover_splits(source_root: Path, single_split_name: str) -> list[str]:
    detected = [split for split in DEFAULT_SPLITS if (source_root / split).is_dir()]
    if detected:
        return detected
    return [single_split_name]


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f'Source root not found: {source_root}')

    _ensure_empty_dir(output_root, args.force)

    splits = _discover_splits(source_root, args.single_split_name)
    split_reports: list[SplitReport] = []
    sequence_reports: list[SequenceReport] = []
    for split in splits:
        split_report, split_sequence_reports = _prepare_split(split, source_root, output_root, args.mode)
        split_reports.append(split_report)
        sequence_reports.extend(split_sequence_reports)

    manifest = {
        'source_root': str(source_root),
        'output_root': str(output_root),
        'mode': args.mode,
        'splits': [asdict(report) for report in split_reports],
        'sequences': [asdict(report) for report in sequence_reports],
    }

    if args.write_manifest:
        manifest_path = output_root / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
