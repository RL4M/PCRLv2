"""

torch中的Dataset定义脚本
"""

import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset

import parameter as para


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, training, ratio=1.0):
        self.crop_size = [128, 128]
        self.training = training
        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(
            map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))
        self.ct_list = self.ct_list[: int(len(self.ct_list) * ratio)]
        self.seg_list = self.seg_list[: int(len(self.seg_list) * ratio)]

    def __getitem__(self, index):
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # min max 归一化
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 在slice平面内随机选取48张slice
        if self.training:
            start_slice = random.randint(0, ct_array.shape[0] - para.size)
            end_slice = start_slice + para.size - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        else:
            start_slice = (ct_array.shape[0] - para.size) // 2
            end_slice = start_slice + para.size - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        # ct_array, seg_array = self.random_crop(ct_array, seg_array)
        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        depth, height, width = x.shape[-3:]
        sx = random.randint(0, height - crop_size[-2] - 1)
        sy = random.randint(0, width - crop_size[-1] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]

        return crop_volume, crop_seg
