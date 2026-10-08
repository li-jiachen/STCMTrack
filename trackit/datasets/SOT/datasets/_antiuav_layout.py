import os
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def construct_antiuav_layout_dataset(constructor, root_path: str, expected_sequences=None):
    """Build a single-object tracking dataset from the Anti-UAV410 on-disk layout.

    Each sequence is a directory under ``root_path`` that contains the frames
    ``000001.jpg, 000002.jpg, ...`` and an ``IR_label.json`` file with the keys
    ``exist`` (per-frame target presence flag) and ``gt_rect`` ([x, y, w, h]).
    ``tools/prepare_antiuav.py`` converts the infrared videos of Anti-UAV to this layout.
    """
    sequence_names = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    sequence_names.sort()

    if expected_sequences is not None and len(sequence_names) != expected_sequences:
        raise ValueError(f'{root_path}: expected {expected_sequences} official-split sequences, found {len(sequence_names)}')
    constructor.set_total_number_of_sequences(len(sequence_names))
    constructor.set_category_id_name_map({0: 'antiuav'})

    for seq_name in sequence_names:
        seq_path = os.path.join(root_path, seq_name)
        json_path = os.path.join(seq_path, 'IR_label.json')

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        exists = np.asarray(data['exist'])
        boxes = np.array(data['gt_rect'], dtype=np.float64)

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if exists.ndim == 0:
            exists = np.array([exists])

        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or len(boxes) != len(exists) or len(boxes) == 0:
            raise ValueError(f'{seq_name}: invalid GT shape/length')
        if not np.isfinite(boxes).all() or not np.isin(exists, (0, 1)).all():
            raise ValueError(f'{seq_name}: non-finite GT or non-binary existence flag')
        if np.any(exists.astype(bool) & ~(boxes[:, 2:] > 0).all(axis=1)):
            raise ValueError(f'{seq_name}: a present target has nonpositive width/height')
        if not exists[0]:
            raise ValueError(f'{seq_name}: the first-frame template needs a present, annotated target')
        expected_images = {f'{i + 1:06d}.jpg' for i in range(len(boxes))}
        actual_images = {name for name in os.listdir(seq_path) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
        if actual_images != expected_images:
            raise ValueError(f'{seq_name}: frame files do not exactly match annotation indices')

        with constructor.new_sequence(category_id=0) as seq_constructor:
            seq_constructor.set_name(seq_name)
            num_frames = len(boxes)
            image_files = [f'{i + 1:06d}.jpg' for i in range(num_frames)]

            for bbox, exist, img_file in zip(boxes, exists, image_files):
                img_path = os.path.join(seq_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as exc:
                    raise ValueError(f'Unreadable frame: {img_path}') from exc

                with seq_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(img_path, (width, height))
                    frame_constructor.set_bounding_box(bbox.tolist(), validity=bool(exist))


def antiuav_cache_identity(root_path):
    """Invalidate cached annotations when a split path, sequence set or labels change."""
    root = Path(root_path).expanduser().resolve()
    digest = hashlib.sha256(str(root).encode('utf-8'))
    for sequence in sorted(path for path in root.iterdir() if path.is_dir()):
        digest.update(sequence.name.encode('utf-8'))
        digest.update(b'\0')
        digest.update((sequence / 'IR_label.json').read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()[:24]
