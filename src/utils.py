import numpy as np
import math
import os
import torch
import random
from PIL import ImageFilter


def get_chest_list(txt_path, data_dir):
    image_names = []
    labels = []
    with open(txt_path, "r") as f:
        for line in f:
            items = line.split()
            image_name = items[0]
            label = items[1:]
            label = [int(i) for i in label]
            image_name = os.path.join(data_dir, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


def get_luna_pretrain_list(ratio):
    x_train = []
    with open("train_val_txt/luna_train.txt", "r") as f:
        for line in f:
            x_train.append(line.strip("\n"))
    return x_train[: int(len(x_train) * ratio)]


def get_luna_finetune_list(ratio, path, train_fold):
    x_train = []
    with open("train_val_txt/luna_train.txt", "r") as f:
        for line in f:
            x_train.append(line.strip("\n"))
    return x_train[int(len(x_train) * ratio) :]


def get_luna_list(config, train_fold, valid_fold, test_fold, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, "subset" + str(i))):
            if suffix in file:
                if file_list is not None and file.split("_")[0] in file_list:
                    x_train.append(os.path.join(config.data, "subset" + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, "subset" + str(i), file))
    for i in valid_fold:
        for file in os.listdir(os.path.join(config.data, "subset" + str(i))):
            if suffix in file:
                x_valid.append(os.path.join(config.data, "subset" + str(i), file))
    for i in test_fold:
        for file in os.listdir(os.path.join(config.data, "subset" + str(i))):
            if suffix in file:
                x_test.append(os.path.join(config.data, "subset" + str(i), file))
    return x_train, x_valid, x_test


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def adjust_learning_rate(epoch, args, optimizer):
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs_list = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs_list.append(int(it))
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs_list))
    # if steps > 0:
    #     new_lr = opt.lr * (opt.lr_decay_rate ** steps)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
