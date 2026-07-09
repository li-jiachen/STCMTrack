import os
import json
import numpy as np
from PIL import Image
from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class ANTIUAV_Train_Seed(BaseSeed):
    def __init__(self, root_path: str = None, data_split='train'):
        if root_path is None:
            # 需要在 consts.yaml 中定义 ANTIUAV410_TRAIN_PATH
            root_path = self.get_path_from_config('ANTIUAV410_TRAIN_PATH')
        super().__init__('ANTIUAV_Train', root_path, data_split, ('train',), 2)

    def construct(self, constructor):
        root_path = self.root_path
        sequence_names = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        sequence_names.sort()

        constructor.set_total_number_of_sequences(len(sequence_names))
        constructor.set_category_id_name_map({0: 'antiuav'})

        for seq_name in sequence_names:
            seq_path = os.path.join(root_path, seq_name)
            json_path = os.path.join(seq_path, 'IR_label.json')
            img_dir = seq_path

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
                image_files = [f'{i+1:06d}.jpg' for i in range(num_frames)]

                for i, (bbox, exist, img_file) in enumerate(zip(boxes, exists, image_files)):
                    img_path = os.path.join(img_dir, img_file)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except:
                        width, height = 0, 0

                    with seq_constructor.new_frame() as frame_constructor:
                        frame_constructor.set_path(img_path, (width, height))
                        validity = bool(exist)
                        frame_constructor.set_bounding_box(bbox.tolist(), validity=validity)