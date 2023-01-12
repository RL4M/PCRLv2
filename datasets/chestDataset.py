import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Pcrlv2ChestPretask(Dataset):
    def __init__(self, config, img_train, train, transform=None, local_transform=None, spatial_transform=None,
                 spatial_transform_local=None, num_local_view=6):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform
        self.spatial_transform = spatial_transform
        self.spatial_transform_local = spatial_transform_local
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.num_local_view = num_local_view
        self.local_transform = local_transform
        self.normalize_trans = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_name = self.imgs[index]
        y = Image.open(image_name).convert('RGB')
        # global
        y1 = self.spatial_transform(y)
        y2 = self.spatial_transform(y)
        norm_y1 = self.normalize_trans(y1)
        norm_y2 = self.normalize_trans(y2)
        x = copy.deepcopy(norm_y1)
        x2 = copy.deepcopy(norm_y2)
        y1 = self.transform(y1)
        y2 = self.transform(y2)
        local_views = []
        for i in range(self.num_local_view):
            local_view = self.spatial_transform_local(y)
            local_view = self.local_transform(local_view)
            local_views.append(local_view)
        return y1, y2, x, x2, local_views


class ChestFineTune(Dataset):
    def __init__(self, imgs, labels, transform=None, train=True):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.image_names = imgs
        self.labels = labels
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels

        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
