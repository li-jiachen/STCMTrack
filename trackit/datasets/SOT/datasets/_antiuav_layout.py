import os
import json
import numpy as np
from PIL import Image


def construct_antiuav_layout_dataset(constructor, root_path: str):
    """Build a single-object tracking dataset from the Anti-UAV410 on-disk layout.

    Each sequence is a directory under ``root_path`` that contains the frames
    ``000001.jpg, 000002.jpg, ...`` and an ``IR_label.json`` file with the keys
    ``exist`` (per-frame target presence flag) and ``gt_rect`` ([x, y, w, h]).
    ``tools/prepare_antiuav.py`` converts the infrared videos of Anti-UAV to this layout.
    """
    sequence_names = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    sequence_names.sort()

    constructor.set_total_number_of_sequences(len(sequence_names))
    constructor.set_category_id_name_map({0: 'antiuav'})

    for seq_name in sequence_names:
        seq_path = os.path.join(root_path, seq_name)
        json_path = os.path.join(seq_path, 'IR_label.json')

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        exists = np.array(data['exist'], dtype=np.int32)
        boxes = np.array(data['gt_rect'], dtype=np.float64)

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if exists.ndim == 0:
            exists = np.array([exists])

        assert len(boxes) == len(exists), f"Length mismatch in {seq_name}"

        with constructor.new_sequence(category_id=0) as seq_constructor:
            seq_constructor.set_name(seq_name)
            num_frames = len(boxes)
            image_files = [f'{i + 1:06d}.jpg' for i in range(num_frames)]

            for bbox, exist, img_file in zip(boxes, exists, image_files):
                img_path = os.path.join(seq_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0

                with seq_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(img_path, (width, height))
                    frame_constructor.set_bounding_box(bbox.tolist(), validity=bool(exist))
