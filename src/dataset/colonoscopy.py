import os
from time import time

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class Colonoscopy(Dataset):
    @staticmethod
    def name2img_name(name):
        return f"{name}.jpg"

    @staticmethod
    def name2mask_name(name):
        return f"{name}_mask.jpg"

    def __init__(self, split_file, img_root, transforms):
        self.img_root = img_root

        # 文件名称
        with open(split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines()]
        self.ids = sorted(self.ids)

        self.transforms = transforms

    def __getitem__(self, index):
        # start = time()
        name = self.ids[index]
        img_t, mask_t = self.get_transformed_img_mask(name)
        # end = time()
        # print(f"加载这张图花了：{end-start}s")
        return img_t, mask_t, index

    def __len__(self):
        return len(self.ids)

    def get_transformed_img_mask(self, name):
        """
        func:
            根据文件名获得 原图和 mask，并且进行 数据增强
        args:
            文件名前缀
        return:
            img: (H, W, C) RGB np.float32
            mask: (H, W)  0-255 -> 0-1 np.uint8
            transforms(img, mask)
        """
        img_path = os.path.join(self.img_root, self.name2img_name(name))
        mask_path = os.path.join(self.img_root, self.name2mask_name(name))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask 的值从 0-255 变成了 0-1
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        mask = mask.astype(np.uint8)

        img_t, mask_t = self.transforms(img, mask)
        return img_t, mask_t

    def get_img(self, index):
        """
        read image and mask, then transform
        :param index:
        :return:
        """
        name = self.ids[index]
        img_path = os.path.join(self.img_root, self.name2img_name(name))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_mask(self, index):
        name = self.ids[index]
        mask_path = os.path.join(self.img_root, self.name2mask_name(name))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        mask = mask.astype(np.uint8)

        return mask

    def get_name(self, index):
        return self.ids[index]


class ColonoscopySingle(Dataset):
    """
    单张图片的数据集
    """

    def __init__(self, img_path, transforms):
        self.img_path = img_path

        # 文件名称
        self.ids = [img_path]
        self.transforms = transforms

    def __getitem__(self, index):
        name = self.ids[index]
        img_t = self.get_transformed_img_mask(name)
        return img_t, index

    def __len__(self):
        return len(self.ids)

    def get_transformed_img_mask(self, name):
        """
        func:
            根据文件名获得 原图和 mask，并且进行 数据增强
        args:
            文件名前缀
        return:
            img: (H, W, C) RGB np.float32
            transforms(img, mask)
        """
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transforms(img)
        return img_t

    def get_img(self, index):
        """
        read image and mask, then transform
        :param index:
        :return:
        """
        name = self.ids[index]
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_name(self, index):
        return self.ids[index]

def collate_fn(batch):
    """
    func:
        从多个 batch 中合并 patch
    args:
        batch: list [(img, mask, index),()...]  len is batch_size
        [0] image: patch, H, W, C
        [1] mask: patch, H, W
        [2] index: int
    return:
        batch_img_t: (batch*patch, C, H, W)
        batch_mask_t: (batch*patch, H, W)
        batch_index: [] list of index, len = batch_size
    """
    batch_img = np.concatenate([b[0] for b in batch], axis=0)
    batch_mask = np.concatenate([b[1] for b in batch], axis=0)
    batch_index = [b[2] for b in batch]

    # N, H, W, C -> N, C, H, W
    batch_img_t = torch.from_numpy(batch_img).permute(0, 3, 1, 2).contiguous().float()
    batch_mask_t = torch.from_numpy(batch_mask)
    return batch_img_t, batch_mask_t, batch_index

def collate_fn_single(batch):
    """
    func:
        从多个 batch 中合并 patch
    args:
        batch: list [(img, mask, index),()...]  len is batch_size
        [0] image: patch, H, W, C
        [2] index: int
    return:
        batch_img_t: (batch*patch, C, H, W)
        batch_index: [] list of index, len = batch_size
    """
    batch_img = np.concatenate([b[0] for b in batch], axis=0)
    batch_index = [b[1] for b in batch]
    # N, H, W, C -> N, C, H, W
    batch_img_t = torch.from_numpy(batch_img).permute(0, 3, 1, 2).contiguous().float()
    return batch_img_t, batch_index