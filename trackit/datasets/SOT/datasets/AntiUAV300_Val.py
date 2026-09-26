from trackit.datasets.common.seed import BaseSeed
from ._antiuav_layout import construct_antiuav_layout_dataset


class AntiUAV300_Val_Seed(BaseSeed):
    """Anti-UAV infrared val split (`antiuav300` in the scripts), prepared with tools/prepare_antiuav.py
    (path constant ``ANTIUAV300_VAL_PATH`` in consts.yaml)."""

    def __init__(self, root_path: str = None, data_split='val'):
        if root_path is None:
            root_path = self.get_path_from_config('ANTIUAV300_VAL_PATH')
        super().__init__('AntiUAV300_Val', root_path, data_split, ('val',), 1)

    def construct(self, constructor):
        construct_antiuav_layout_dataset(constructor, self.root_path)
