import numpy as np
import math
import os
import torch
import random
from PIL import ImageFilter
import torch.nn.functional as F


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
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[:int(len(x_train) * ratio)]


def get_luna_finetune_list(ratio, path, train_fold):
    x_train = []
    with open('train_val_txt/luna_train.txt', 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
    return x_train[int(len(x_train) * ratio):]


def get_luna_list(config, train_fold, valid_fold, test_fold, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                if file_list is not None and file.split('_')[0] in file_list:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in valid_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_valid.append(os.path.join(config.data, 'subset' + str(i), file))
    for i in test_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                x_test.append(os.path.join(config.data, 'subset' + str(i), file))
    return x_train, x_valid, x_test


def get_brats_list(data, ratio):
    val_patients_list = []
    train_patients_list = []
    test_patients_list = []
    with open('./train_val_txt/brats_train.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            train_patients_list.append(os.path.join(data, line))
    with open('./train_val_txt/brats_valid.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            val_patients_list.append(os.path.join(data, line))
    with open('./train_val_txt/brats_test.txt', 'r') as f:

        for line in f:
            line = line.strip('\n')
            test_patients_list.append(os.path.join(data, line))
    train_patients_list = train_patients_list[: int(len(train_patients_list) * ratio)]
    print(
        f"train patients: {len(train_patients_list)}, valid patients: {len(val_patients_list)},"
        f"test patients {len(test_patients_list)}")
    return train_patients_list, val_patients_list, test_patients_list


def get_luna_finetune_nodule(config, train_fold, valid_txt, test_txt, suffix, file_list):
    x_train = []
    x_valid = []
    x_test = []
    for i in train_fold:
        for file in os.listdir(os.path.join(config.data, 'subset' + str(i))):
            if suffix in file:
                if file_list is not None and file.split('_')[0] in file_list:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
                elif file_list is None:
                    x_train.append(os.path.join(config.data, 'subset' + str(i), file))
    with open(valid_txt, 'r') as f:
        for line in f:
            x_valid.append(line.strip('\n'))
    with open(test_txt, 'r') as f:
        for line in f:
            x_test.append(line.strip('\n'))
    return x_train, x_valid, x_test


def divide__luna_true_positive(data_list):
    true_list = []
    false_list = []
    for i in data_list:
        name = os.path.split(i)[-1]
        label = name.split('_')[1]
        if label == '1':
            true_list.append(i)
        else:
            false_list.append(i)
    return true_list, false_list


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

            mask[y1: y2, x1: x2] = 0.

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
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def bceDiceLoss(input, target, train=True):
    bce = F.binary_cross_entropy_with_logits(input, target)
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    if train:
        return dice + 0.2 * bce
    return dice


def thor_dice_loss(input, target, train=True):
    # print(input.shape, target.shape)
    es_dice = bceDiceLoss(input[:, 0], target[:, 0], train)
    tra_dice = bceDiceLoss(input[:, 1], target[:, 1], train)
    aor_dice = bceDiceLoss(input[:, 2], target[:, 2], train)
    heart_dice = bceDiceLoss(input[:, 3], target[:, 3], train)
    print(f'label1 dice {es_dice}, label2 dice {tra_dice}, label3 dice{aor_dice}, label4 dice{heart_dice}')
    return es_dice + tra_dice + aor_dice + heart_dice


def brats_dice_loss(input, target, train=True):
    wt_loss = bceDiceLoss(input[:, 0], target[:, 0], train)
    tc_loss = bceDiceLoss(input[:, 1], target[:, 1], train)
    et_loss = bceDiceLoss(input[:, 2], target[:, 2], train)
    print(f'wt loss: {wt_loss}, tc_loss : {tc_loss}, et_loss: {et_loss}')
    return wt_loss + tc_loss + et_loss
