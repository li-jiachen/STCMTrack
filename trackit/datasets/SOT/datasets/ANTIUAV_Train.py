from trackit.datasets.common.seed import BaseSeed
from ._antiuav_layout import construct_antiuav_layout_dataset


class ANTIUAV_Train_Seed(BaseSeed):
    """Anti-UAV410 training split (path constant ``ANTIUAV410_TRAIN_PATH`` in consts.yaml)."""

    def __init__(self, root_path: str = None, data_split='train'):
        if root_path is None:
            root_path = self.get_path_from_config('ANTIUAV410_TRAIN_PATH')
        super().__init__('ANTIUAV_Train', root_path, data_split, ('train',), 2)

    def construct(self, constructor):
        construct_antiuav_layout_dataset(constructor, self.root_path)
