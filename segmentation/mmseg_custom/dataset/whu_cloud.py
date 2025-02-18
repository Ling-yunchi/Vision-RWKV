from mmseg.datasets import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class WhuCloudDataset(CustomDataset):

    CLASSES = ("none", "cloud", "shadow")

    PALETTE = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(WhuCloudDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
