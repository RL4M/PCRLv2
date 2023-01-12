from torch.utils.data import DataLoader

from datasets import *
from utils import *
from torchvision import transforms, datasets
import torch
import torchio.transforms
import copy


class DataGenerator:

    def __init__(self, args):
        self.args = args

    def pcrlv2_chest_pretask(self):
        args = self.args
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        spatial_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()
        ])
        spatial_transform_local = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.05, 0.3)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()

        ])
        train_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
        local_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform.transforms.append(Cutout(n_holes=3, length=32))
        train_file = './train_val_txt/chest_train.txt'
        train_imgs, train_labels = get_chest_list(train_file, args.data)
        train_imgs = train_imgs[:int(len(train_imgs) * args.ratio)]
        train_dataset = Pcrlv2ChestPretask(args, train_imgs, transform=train_transform,
                                           local_transform=local_transform, train=True,
                                           spatial_transform=spatial_transform,
                                           spatial_transform_local=spatial_transform_local, num_local_view=6)
        print(len(train_dataset))
        train_sampler = None
        dataloader = {}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.b, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        dataloader['train'] = train_loader
        dataloader['eval'] = train_loader

        return dataloader

    def pcrlv2_luna_pretask(self):
        print('using the reverse_aug pretrain on luna')
        args = self.args
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6]
        valid_fold = [7, 8, 9]
        file_list = get_luna_pretrain_list(args.ratio)
        x_train, x_valid, _ = get_luna_list(args, train_fold, valid_fold, valid_fold, suffix='_global_',
                                            file_list=file_list)
        print(f'total train images {len(x_train)}, valid images {len(x_valid)}')
        spatial_transforms = [torchio.transforms.RandomFlip(),
                              torchio.transforms.RandomAffine(),
                              ]
        spatial_transforms = torchio.transforms.Compose(spatial_transforms)
        transforms = [torchio.transforms.RandomBlur(),
                      torchio.transforms.RandomNoise(),
                      torchio.transforms.RandomGamma(),
                      torchio.transforms.ZNormalization()
                      ]
        local_transforms = torchio.transforms.Compose(transforms)
        global_transforms = [torchio.transforms.RandomBlur(),
                             torchio.transforms.RandomNoise(),
                             torchio.transforms.RandomGamma(),
                             torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
                             torchio.transforms.ZNormalization()
                             ]
        global_transforms = torchio.transforms.Compose(global_transforms)

        train_ds = Pcrlv2LunaPretask(args, x_train, train=True, transform=spatial_transforms,
                                     global_transforms=global_transforms, local_transforms=local_transforms)
        valid_ds = Pcrlv2LunaPretask(args, x_valid, train=False)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers)
        return dataloader

    def pcrlv2_chest_finetune(self):
        args = self.args
        train_file = './train_val_txt/chest_train.txt'
        valid_file = './train_val_txt/chest_valid.txt'
        test_file = './train_val_txt/chest_test.txt'
        dataloader = {}
        train_imgs, train_labels = get_chest_list(train_file, args.data)
        valid_imgs, valid_labels = get_chest_list(valid_file, args.data)
        test_imgs, test_labels = get_chest_list(test_file, args.data)
        transCrop = 224
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.RandomRotation(10))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        train_transform = transforms.Compose(transformList)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        testResize = 256
        test_transformList = []
        test_transformList.append(transforms.Resize(testResize))
        test_transformList.append(transforms.CenterCrop(224))
        test_transformList.append(transforms.ToTensor())
        test_transformList.append(normalize)
        test_transform = transforms.Compose(test_transformList)
        train_imgs = train_imgs[int(len(train_imgs) * args.ratio):]
        train_labels = train_labels[int(len(train_labels) * args.ratio):]
        print(len(train_imgs), len(valid_imgs), len(test_imgs))
        train_ds = ChestFineTune(labels=train_labels, imgs=train_imgs, transform=train_transform, train=True)
        valid_ds = ChestFineTune(labels=valid_labels, imgs=valid_imgs, transform=train_transform, train=False)
        test_ds = ChestFineTune(labels=test_labels, imgs=test_imgs, transform=test_transform, train=False)
        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers)
        dataloader['test'] = DataLoader(test_ds, batch_size=args.b, pin_memory=True, shuffle=False,
                                        num_workers=args.workers)
        return dataloader

    def pcrlv2_brats_finetune(self):
        dataloader = {}
        train_list, val_list, test_list = get_brats_list(self.args.data, self.args.ratio)
        train_ds = BratsFinetune(train_list, train=True)
        val_ds = BratsFinetune(val_list, train=False)
        test_ds = BratsFinetune(test_list, train=False)
        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(val_ds, batch_size=self.args.b,
                                        num_workers=self.args.workers,
                                        pin_memory=True,
                                        shuffle=False)
        dataloader['test'] = DataLoader(test_ds, batch_size=1, num_workers=self.args.b,
                                        pin_memory=True,
                                        shuffle=False)
        return dataloader

    def pcrlv2_luna_finetune(self):
        luna_valid_txt = 'train_val_txt/luna_finetune_test.txt'
        luna_test_txt = 'train_val_txt/luna_finetune_valid.txt'
        # luna_train_txt = 'train_val_txt/luna_finetune_train.txt'
        args = self.args
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6]
        valid_fold = [7, 8]
        test_fold = [9]
        file_list = get_luna_finetune_list(args.ratio, args.data, train_fold)
        print(len(file_list))
        x_train, x_valid, x_test = get_luna_finetune_nodule(args, train_fold, luna_valid_txt, luna_test_txt,
                                                            suffix='.npy',
                                                            file_list=file_list)
        print(len(x_train))
        train_ds_true, train_ds_false = divide__luna_true_positive(x_train)
        valid_ds_true, valid_ds_false = divide__luna_true_positive(x_valid)
        test_ds_true, test_ds_false = divide__luna_true_positive(x_test)

        train_ds = LunaFineTune(args, train_ds_true, train_ds_false, train=True)
        valid_ds = LunaFineTune(args, valid_ds_true, valid_ds_false, train=False)
        test_ds = LunaFineTune(args, test_ds_true, test_ds_false, train=False, test=True)
        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers)
        dataloader['test'] = DataLoader(test_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers)
        print(f"training nodule{len(train_ds)} valid noudle {len(valid_ds)}, test noudle {len(test_ds)}")
        return dataloader
