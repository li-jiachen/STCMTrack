from trackit.datasets.common.seed import BaseSeed
from ._antiuav_layout import construct_antiuav_layout_dataset


class ANTIUAV_Val_Seed(BaseSeed):
    """Anti-UAV410 validation split (path constant ``ANTIUAV410_VAL_PATH`` in consts.yaml)."""

    def __init__(self, root_path: str = None, data_split='val'):
        if root_path is None:
            root_path = self.get_path_from_config('ANTIUAV410_VAL_PATH')
        super().__init__('ANTIUAV_Val', root_path, data_split, ('val',), 2)

    def construct(self, constructor):
        construct_antiuav_layout_dataset(constructor, self.root_path)
