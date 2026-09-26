from trackit.datasets.common.seed import BaseSeed
from ._antiuav_layout import construct_antiuav_layout_dataset


class AntiUAV300_Seed(BaseSeed):
    """Anti-UAV infrared test split (`antiuav300` in the scripts), prepared with tools/prepare_antiuav.py
    (path constant ``ANTIUAV300_TEST_PATH`` in consts.yaml)."""

    def __init__(self, root_path: str = None, data_split='test'):
        if root_path is None:
            root_path = self.get_path_from_config('ANTIUAV300_TEST_PATH')
        super().__init__('AntiUAV300', root_path, data_split, ('test',), 1)

    def construct(self, constructor):
        construct_antiuav_layout_dataset(constructor, self.root_path)
