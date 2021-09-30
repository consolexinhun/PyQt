import random

import cv2
import numpy as np

from registry import TRANSFORM
from lib.post_process import valid_patch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, mask=None):
        for t in self.transforms:
            if mask is not None:
                img, mask = t(img, mask)
            else:
                img = t(img)
        if mask is not None:
            return img,  mask
        else:
            return img
class Convert2Int:
    """
    func:
        将 img, mask 转为整数
        img: 0-255
        mask: 0-num_class
    return:
        img, mask: np.uint8
    """
    def __call__(self, img, mask=None):
        if mask is not None:
            return img.astype(np.uint8), mask
        return img.astype(np.uint8)


class RandomDownSample:
    """
    func:
        双线性插值下采样
    return:
        shape/downsample_rate
    """
    def __init__(self, downsample_rate=1.0):
        self.scale = 1.0/downsample_rate
    def __call__(self, img, mask=None):
        scale = self.scale

        # 双线性插值， dsize 设为0将会根据 scale 推断输出图像的大小
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            mask = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            return img, mask
        else:
            return img

@TRANSFORM.register("RandomSampleCrop")
class RandomSampleCrop:
    """
    func:
        随机裁剪，如果没有满足 crop_size ，那么填充 0
    parmas:
        img: (H, W, C)
        mask: (H, W)
    return:
        img_patches: (crop_nums, crop_size, crop_size, 3)
        mask_patches: (crop_nums, crop_size, crop_size)

    """
    def __init__(self, **kwargs):
        self.crop_size = kwargs["crop_size"]
        self.crop_nums = kwargs["crop_nums"]

    def get_random_patch(self, img, mask):
        # 有可能 img 比 crop_size 要小，毕竟是下采样之后的
        while True:
            h, w = img.shape[:2]
            if self.crop_size >= w:
                x1 = 0
                x2 = w
            else:
                x1 = random.randint(0, w - self.crop_size)
                x2 = x1 + self.crop_size

            if self.crop_size >= h:
                y1 = 0
                y2 = h
            else:
                y1 = random.randint(0, h - self.crop_size)
                y2 = y1 + self.crop_size

            img_patch = img[y1:y2, x1:x2]
            if valid_patch(img_patch):
                mask_patch = mask[y1:y2, x1:x2]
                return img_patch, mask_patch

    def __call__(self, img, mask):
        crop_nums = self.crop_nums
        crop_size = self.crop_size

        img_patches = np.zeros((crop_nums, crop_size, crop_size, 3)).astype(np.uint8)
        mask_patches = np.zeros((crop_nums, crop_size, crop_size)).astype(np.uint8)

        for i in range(crop_nums):
            img_patch, mask_patch = self.get_random_patch(img, mask)
            h, w = img.shape[:2]
            img_patches[i, :h, :w, :] = img_patch
            mask_patches[i, :h, :w] = mask_patch
        return img_patches, mask_patches


@TRANSFORM.register("ConcatRandomSampleCrop")
class ConcatRandomSampleCrop:
    """
    func:
        随机裁剪，如果没有满足 crop_size ，那么填充 0
    parmas:
        img: (H, W, C)
        mask: (H, W)
    return:
        img_patches: (crop_nums, crop_size, crop_size, 6)
        mask_patches: (crop_nums, crop_size, crop_size)

    """
    def __init__(self, **kwargs):
        self.crop_size = kwargs["crop_size"]
        self.crop_nums = kwargs["crop_nums"]

    def get_random_patch(self, img, mask):
        # 有可能 img 比 crop_size 要小，毕竟是下采样之后的
        while True:
            h, w = img.shape[:2]
            if self.crop_size >= w:
                x1 = 0
                x2 = w
            else:
                x1 = random.randint(0, w - self.crop_size)
                x2 = x1 + self.crop_size

            if self.crop_size >= h:
                y1 = 0
                y2 = h
            else:
                y1 = random.randint(0, h - self.crop_size)
                y2 = y1 + self.crop_size

            img_patch = img[y1:y2, x1:x2]
            if valid_patch(img_patch):
                mask_patch = mask[y1:y2, x1:x2]
                return img_patch, mask_patch

    def __call__(self, img, mask):
        crop_nums = self.crop_nums
        crop_size = self.crop_size

        img_patches = np.zeros((crop_nums, crop_size, crop_size, 6)).astype(np.uint8)
        mask_patches = np.zeros((crop_nums, crop_size, crop_size)).astype(np.uint8)

        global_img = cv2.resize(img, dsize=(crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        for i in range(crop_nums):
            img_patch, mask_patch = self.get_random_patch(img, mask)
            h, w = img.shape[:2]

            img_patches[i, :h, :w, :3] = img_patch
            img_patches[i, :, :, 3:] = global_img

            mask_patches[i, :h, :w] = mask_patch
        return img_patches, mask_patches

@TRANSFORM.register("Resize")
class Resize:
    def __init__(self, **kwargs):
        self.size = kwargs["crop_size"]

    def __call__(self, img, mask=None):
        img = self._resize(img, self.size)
        if mask is not None:
            mask = self._resize(mask, self.size)
            return img[None, :, :, :], mask[None, :, :]
        else:
            return img

    def _resize(self, img, size):
        h, w = img.shape[:2]
        # scale = max(h/size, w/size)
        scale = min(size/h, size/w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_h, new_w = img.shape[:2]

        if len(img.shape) == 3:  # img
            new_img = np.zeros((size, size, 3))
        elif len(img.shape) == 2:  # mask
            new_img = np.zeros((size, size))
        else :
            raise Exception("形状不对")

        padding_h, padding_w = (size - new_h)//2, (size - new_w)//2
        new_img[padding_h: padding_h+new_h, padding_w: padding_w+new_w] = img
        return new_img


class RandomRotate:
    """
    func:
        随机旋转
    params:
        img: (B, H, W, C)
        mask: (H, W, C)
    """
    def __call__(self, img, mask):
        for i in range(img.shape[0]):
            rotate_times = random.randint(0, 3)
            img[i] = np.ascontiguousarray(np.rot90(img[i], rotate_times))
            mask[i] = np.ascontiguousarray(np.rot90(mask[i], rotate_times))

        return img, mask

class RandomHorizontalFlip:
    """
    func:
        随机概率水平翻转
    parmas:
        img: (B, H, W, C)
        mask: (B, H, W, C)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        for i in range(img.shape[0]):
            if random.random() < self.p:
                img[i] = np.ascontiguousarray(img[i, ::-1, :])
                mask[i] = np.ascontiguousarray(mask[i, ::-1, :])
        return img, mask

class RandomVerticalFlip:
    """
    func:
        随机概率垂直翻转
    parmas:
        img: (B, H, W, C)
        mask: (B, H, W, C)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        for i in range(img.shape[0]):
            if random.random() < self.p:
                img[i] = np.ascontiguousarray(img[i, :, ::-1])
                mask[i] = np.ascontiguousarray(mask[i, :, ::-1])

        return img, mask

@TRANSFORM.register("DenseCrop")
class DenseCrop:
    """
    func:
        测试时用的密集裁剪
    params:
        img: (H, W, C)
        mask: (H, W)
    return:
        img_patches: (crop_xys), crop_size, crop_size, 3) np.uint8
        mask_patches: (crop_xys), crop_size, crop_size) np.uint8
    """
    def __init__(self, **kwargs):
        self.crop_size = kwargs["crop_size"]
        self.crop_stride = kwargs["crop_stride"]

    def get_grid(self, img_size):
        """
        func:
            带有重叠的补丁裁剪
            和 后处理中的一模一样
        return:
            得到补丁在大图上的横纵坐标
        """
        res = []
        cur = 0
        while True:
            x2 = min(cur+self.crop_size, img_size)
            x1 = max(0, x2-self.crop_size)
            res.append([x1, x2])

            if x2 == img_size:
                break
            cur += self.crop_stride
        return res
    def get_xys(self, shape):
        h, w = shape[:2]
        h_grid = self.get_grid(h)
        w_grid = self.get_grid(w)
        xy_grid = [[x[0], y[0], x[1], y[1]] for y in h_grid for x in w_grid]

        return xy_grid

    def __call__(self, img, mask=None):
        crop_xys = self.get_xys(img.shape)  # 总共多少个补丁
        crop_size = self.crop_size

        img_patches = np.zeros((len(crop_xys), crop_size, crop_size, 3), dtype=np.uint8)
        mask_patches = np.zeros((len(crop_xys), crop_size, crop_size), dtype=np.uint8)

        for i, xy in enumerate(crop_xys):
            x1, y1, x2, y2 = xy
            img_patches[i, :y2-y1, :x2-x1, :] = img[y1:y2, x1:x2]

            if mask is not None:
                mask_patches[i, :y2-y1, :x2-x1] = mask[y1:y2, x1:x2]
        if mask is not None:
            return img_patches, mask_patches
        else:
            return img_patches




@TRANSFORM.register("ConcatDenseCrop")
class ConcatDenseCrop:
    """
    func:
        测试时用的密集裁剪
    params:
        img: (H, W, C)
        mask: (H, W)
    return:
        img_patches: (crop_xys), crop_size, crop_size, 6) np.uint8
        mask_patches: (crop_xys), crop_size, crop_size) np.uint8
    """
    def __init__(self, **kwargs):
        self.crop_size = kwargs["crop_size"]
        self.crop_stride = kwargs["crop_stride"]

    def get_grid(self, img_size):
        """
        func:
            带有重叠的补丁裁剪
            和 后处理中的一模一样
        return:
            得到补丁在大图上的横纵坐标
        """
        res = []
        cur = 0
        while True:
            x2 = min(cur+self.crop_size, img_size)
            x1 = max(0, x2-self.crop_size)
            res.append([x1, x2])

            if x2 == img_size:
                break
            cur += self.crop_stride
        return res
    def get_xys(self, shape):
        h, w = shape[:2]
        h_grid = self.get_grid(h)
        w_grid = self.get_grid(w)
        xy_grid = [[x[0], y[0], x[1], y[1]] for y in h_grid for x in w_grid]

        return xy_grid

    def __call__(self, img, mask=None):
        crop_xys = self.get_xys(img.shape)  # 总共多少个补丁
        crop_size = self.crop_size

        img_patches = np.zeros((len(crop_xys), crop_size, crop_size, 6), dtype=np.uint8)
        mask_patches = np.zeros((len(crop_xys), crop_size, crop_size), dtype=np.uint8)

        global_img = cv2.resize(img, dsize=(crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        for i, xy in enumerate(crop_xys):
            x1, y1, x2, y2 = xy
            img_patches[i, :y2-y1, :x2-x1, :3] = img[y1:y2, x1:x2]
            img_patches[i, :, :, 3:] = global_img

            if mask is not None:
                mask_patches[i, :y2-y1, :x2-x1] = mask[y1:y2, x1:x2]
        if mask is not None:
            return img_patches, mask_patches
        else:
            return img_patches

class ConcatNormalize:
    """
    func:
        标准化，减去均值除以方差
    return:
        img: np.float32
        mask: np.uint8
    """
    def __init__(self, mean, std):
        self.mean = np.tile(np.float32(mean), 2)  # [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
        self.std = np.tile(np.float32(std), 2)  # repeat: [1, 2, 3] -> [1, 1, 2, 2, 3, 3]

    def __call__(self, img, mask=None):
        img = ((np.float32(img) - self.mean)/self.std)
        if mask is not None:
            return img, mask
        else:
            return img

class Normalize:
    """
    func:
        标准化，减去均值除以方差
    return:
        img: np.float32
        mask: np.uint8
    """
    def __init__(self, mean, std):
        self.mean = np.float32(mean)
        self.std = np.float32(std)

    def __call__(self, img, mask=None):
        img = ((np.float32(img) - self.mean)/self.std)
        if mask is not None:
            return img, mask
        else:
            return img
