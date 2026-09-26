#!/usr/bin/env python3
"""Convert the infrared videos of Anti-UAV (`antiuav300` in the scripts) to the layout used by STCMTrack.

Output layout (identical to Anti-UAV410, so the same loader and metric script apply):

    <output-root>/<split>/<sequence>/000001.jpg, 000002.jpg, ...
    <output-root>/<split>/<sequence>/IR_label.json   # {"exist": [...], "gt_rect": [[x, y, w, h], ...]}

Each source sequence directory is expected to contain one infrared annotation file
(``infrared.json`` or ``IR_label.json``) and either an infrared video
(``infrared.mp4`` or ``IR.mp4``) or already extracted frames. Frames without a valid
box are written as ``[0, 0, 0, 0]`` with ``exist = 0``; the metric script skips them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

ANNOTATION_NAMES = ('infrared.json', 'IR_label.json')
VIDEO_NAMES = ('infrared.mp4', 'IR.mp4')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
SPLIT_ALIASES = {
    'train': ('train', 'Train', 'training'),
    'val': ('val', 'Val', 'validation', 'Validation'),
    'test': ('test', 'Test'),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare the infrared videos of Anti-UAV for STCMTrack.')
    parser.add_argument('--source-root', required=True, help='Anti-UAV root containing train/val/test sequence folders.')
    parser.add_argument('--output-root', required=True, help='Where the converted dataset is written.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--gt-format', choices=('xywh', 'xyxy'), default='xywh',
                        help='Box format of gt_rect in the source annotation files.')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--force', action='store_true', help='Overwrite sequences that already exist in the output.')
    return parser.parse_args()


def _find_split_dir(source_root: Path, split: str) -> Path:
    for name in SPLIT_ALIASES.get(split, (split,)):
        if (source_root / name).is_dir():
            return source_root / name
    raise FileNotFoundError(f'Split "{split}" not found under {source_root}')


def _first_existing(directory: Path, names) -> Path | None:
    for name in names:
        if (directory / name).exists():
            return directory / name
    return None


def _normalize_annotation(data: dict, gt_format: str) -> tuple[list[int], list[list[float]]]:
    boxes = data.get('gt_rect', [])
    exists = data.get('exist', [1] * len(boxes))
    if len(boxes) != len(exists):
        raise ValueError(f'gt_rect ({len(boxes)}) and exist ({len(exists)}) have different lengths')
    out_exists, out_boxes = [], []
    for box, exist in zip(boxes, exists):
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, a, b = (float(v) for v in box)
            w, h = (a - x1, b - y1) if gt_format == 'xyxy' else (a, b)
            if int(exist) and w > 0 and h > 0:
                out_exists.append(1)
                out_boxes.append([x1, y1, w, h])
                continue
        out_exists.append(0)
        out_boxes.append([0.0, 0.0, 0.0, 0.0])
    return out_exists, out_boxes


def _extract_video(video_path: Path, target_dir: Path, num_frames: int, jpeg_quality: int) -> int:
    capture = cv2.VideoCapture(str(video_path))
    written = 0
    while written < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        written += 1
        cv2.imwrite(str(target_dir / f'{written:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    capture.release()
    return written


def _copy_frames(sequence_dir: Path, target_dir: Path, num_frames: int) -> int:
    images = sorted(p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for index, image_path in enumerate(images[:num_frames], start=1):
        target = target_dir / f'{index:06d}.jpg'
        if image_path.suffix.lower() in ('.jpg', '.jpeg'):
            shutil.copyfile(image_path, target)
        else:
            cv2.imwrite(str(target), cv2.imread(str(image_path), cv2.IMREAD_COLOR))
    return min(len(images), num_frames)


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = {'source_root': str(source_root), 'output_root': str(output_root), 'splits': {}}

    for split in args.splits:
        split_dir = _find_split_dir(source_root, split)
        sequences = sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith('.'))
        split_report = {'sequences': 0, 'frames': 0, 'valid_boxes': 0, 'short_sequences': []}
        for sequence_dir in sequences:
            annotation_path = _first_existing(sequence_dir, ANNOTATION_NAMES)
            if annotation_path is None:
                raise FileNotFoundError(f'No infrared annotation ({", ".join(ANNOTATION_NAMES)}) in {sequence_dir}')
            with annotation_path.open('r', encoding='utf-8') as f:
                exists, boxes = _normalize_annotation(json.load(f), args.gt_format)

            target_dir = output_root / split / sequence_dir.name
            if target_dir.exists():
                if not args.force:
                    raise FileExistsError(f'{target_dir} exists; use --force to overwrite')
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True)

            video_path = _first_existing(sequence_dir, VIDEO_NAMES)
            if video_path is not None:
                num_written = _extract_video(video_path, target_dir, len(boxes), args.jpeg_quality)
            else:
                num_written = _copy_frames(sequence_dir, target_dir, len(boxes))
            if num_written < len(boxes):
                split_report['short_sequences'].append(sequence_dir.name)
                exists, boxes = exists[:num_written], boxes[:num_written]

            with (target_dir / 'IR_label.json').open('w', encoding='utf-8') as f:
                json.dump({'exist': exists, 'gt_rect': boxes}, f)
            split_report['sequences'] += 1
            split_report['frames'] += len(boxes)
            split_report['valid_boxes'] += sum(exists)
        manifest['splits'][split] = split_report
        print(f'{split}: {split_report["sequences"]} sequences, {split_report["frames"]} frames, '
              f'{split_report["valid_boxes"]} valid boxes', flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + os.linesep, encoding='utf-8')


if __name__ == '__main__':
    main()
