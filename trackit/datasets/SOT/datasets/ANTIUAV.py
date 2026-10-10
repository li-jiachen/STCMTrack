import os
from trackit.datasets.common.seed import BaseSeed
from ._antiuav_layout import construct_antiuav_layout_dataset, antiuav_cache_identity


class ANTIUAV_Seed(BaseSeed):
    

    def __init__(self, root_path: str = None, data_split='test'):
        if root_path is None:
            root_path = os.environ.get('ANTIUAV_GT_DIR') or self.get_path_from_config('ANTIUAV410_PATH')
        super().__init__('ANTIUAV410', root_path, data_split, ('test',), 3)

        self.cache_identity = antiuav_cache_identity(self.root_path)

    def construct(self, constructor):
        construct_antiuav_layout_dataset(constructor, self.root_path, expected_sequences=120)
