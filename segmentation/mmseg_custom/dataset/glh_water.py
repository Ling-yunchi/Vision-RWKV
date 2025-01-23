from mmseg.datasets import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class GLHWaterDataset(CustomDataset):

    CLASSES = ("non-water", "water")

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(GLHWaterDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)