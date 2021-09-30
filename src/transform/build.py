from .transforms import Convert2Int, RandomDownSample, RandomRotate, RandomHorizontalFlip,\
     RandomVerticalFlip, Normalize, Compose
# 未显式调用，在参数中指明
from .transforms import RandomSampleCrop, DenseCrop, Resize, ConcatRandomSampleCrop, ConcatDenseCrop, ConcatNormalize

from registry import TRANSFORM

def build_transforms(cfg, is_train=True):
    if is_train:
        """
        训练：
            双线性插值下采样，变成原来的 1/r
            随机裁剪
            旋转，水平垂直翻转
        """
        transforms = [
            Convert2Int(),
            RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE),
            TRANSFORM[cfg.DATA.CROP_METHOD_TRAIN](
                crop_size=cfg.DATA.SIZE_TRAIN,
                crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN
            ),  # 裁剪SampleCrop或者是Resize
            RandomRotate() if cfg.DATA.RANDOM_ROTATE else None,
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]   
    else:
        """
        验证和测试：
            双线性插值下采样，变成原来的 1/r
            密集滑动窗口裁剪
        """
        transforms = [
            Convert2Int(),
            RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE_TEST),
            TRANSFORM[cfg.DATA.CROP_METHOD_TEST](
                crop_size=cfg.DATA.SIZE_TEST,
                crop_stride=cfg.DATA.DENSE_CROP_STRIDE
            ),
            Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]
    transforms = Compose(transforms)
    return transforms

def global_build_transforms(cfg, is_train=True):
    if is_train:
        """
        训练：
            双线性插值下采样，变成原来的 1/r
            随机裁剪
            旋转，水平垂直翻转

        """
        transforms = [
            Convert2Int(),
            # RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE),
            TRANSFORM[cfg.DATA.CROP_METHOD_TRAIN](
                crop_size=cfg.DATA.SIZE_TRAIN,
                crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN
            ),  # 裁剪或者是Resize
            RandomRotate() if cfg.DATA.RANDOM_ROTATE else None,
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]   
    else:
        """
        验证和测试：
            双线性插值下采样，变成原来的 1/r
            密集滑动窗口裁剪
        """
        transforms = [
            Convert2Int(),
            # RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE_TEST),
            TRANSFORM[cfg.DATA.CROP_METHOD_TEST](
                crop_size=cfg.DATA.SIZE_TEST,
                crop_stride=cfg.DATA.DENSE_CROP_STRIDE
            ),
            Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]
    transforms = Compose(transforms)
    return transforms


def concat_build_transforms(cfg, is_train=True):
    if is_train:
        """
        训练：
            双线性插值下采样，变成原来的 1/r
            随机裁剪
            旋转，水平垂直翻转
        """
        transforms = [
            Convert2Int(),
            RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE),
            ConcatRandomSampleCrop(
                crop_size=cfg.DATA.SIZE_TRAIN,
                crop_nums=cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN),
            RandomRotate() if cfg.DATA.RANDOM_ROTATE else None,
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ConcatNormalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]   
    else:
        """
        验证和测试：
            双线性插值下采样，变成原来的 1/r
            密集滑动窗口裁剪
        """
        transforms = [
            Convert2Int(),
            RandomDownSample(downsample_rate=cfg.DATA.DOWN_SAMPLE_RATE_TEST),
            ConcatDenseCrop(
                crop_size=cfg.DATA.SIZE_TEST,
                crop_stride=cfg.DATA.DENSE_CROP_STRIDE),
            ConcatNormalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        ]
    transforms = Compose(transforms)
    return transforms


