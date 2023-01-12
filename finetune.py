import os
import sys

import numpy as np
import torch
import torch.nn as nn
import random
import segmentation_models_pytorch as smp
from utils import brats_dice_loss
from models.pcrlv2_model_3d import PCRLv23d, SegmentationModel


def train_chest_classification(args, dataloader):
    criterion = nn.BCELoss()
    train_generator = dataloader['train']
    valid_generator = dataloader['eval']
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.2,  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=14,  # define number of output labels
    )
    model = smp.Unet('resnet18', in_channels=3, aux_params=aux_params, classes=1, encoder_weights=None)
    if args.weight is not None:
        weight_path = args.weight
        encoder_dict = torch.load(weight_path)['state_dict']
        encoder_dict['fc.bias'] = 0
        encoder_dict['fc.weight'] = 0
        model.encoder.load_state_dict(encoder_dict)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0
    for epoch in range(0, args.epochs + 1):
        scheduler.step(epoch)
        model.train()
        for iteration, (image, gt) in enumerate(train_generator):
            image = image.cuda().float()
            gt = gt.cuda().float()
            _ , pred = model(image)
            # print(pred.shape, gt.shape)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 5 == 0:
                print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                      .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
                sys.stdout.flush()
        with torch.no_grad():
            model.eval()
            print("validating....")
            for i, (x, y) in enumerate(valid_generator):
                x = x.cuda()
                y = y.cuda().float()
                _, pred = model(x)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

        # logging
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),  # only save encoder
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output,
                            args.model + "_" + args.n + '_' + args.phase + '_' + str(args.ratio) + '.pt'))
            print("Saving model ", os.path.join(args.output, args.model + "_" + args.n + '_' + args.phase + '_' + str(
                args.ratio) + '.pt'))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Ea`rly Stopping")
                break
        sys.stdout.flush()


def train_brats_segmentation(args, dataloader):
    criterion = brats_dice_loss
    train_generator = dataloader['train']
    valid_generator = dataloader['eval']

    model = SegmentationModel(in_channels=4, n_class=3, norm='gn')

    # model = PCRLv23d(n_class=3, norm='bn') # use gn
    weight_path = args.weight
    model_dict = model.state_dict()
    state_dict = torch.load(weight_path)['state_dict']
    first_conv_weight = state_dict['down_tr64.ops.0.conv1.weight']
    first_conv_weight = first_conv_weight.repeat((1, 4, 1, 1, 1))
    state_dict['down_tr64.ops.0.conv1.weight'] = first_conv_weight
    pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'down_tr' in k} # only load the encoder part
    model_dict.update(pretrain_dict)
    print(pretrain_dict.keys())
    model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0
    for epoch in range(0, args.epochs + 1):
        scheduler.step(epoch)
        model.train()
        for iteration, (image, gt) in enumerate(train_generator):
            image = image.cuda().float()
            gt = gt.cuda().float()
            pred = model(image)
            # print(pred.shape, gt.shape)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 5 == 0:
                print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                      .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
                sys.stdout.flush()
        with torch.no_grad():
            model.eval()
            print("validating....")
            for i, (x, y) in enumerate(valid_generator):
                x = x.cuda()
                y = y.cuda().float()
                pred = model(x)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

        # logging
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),  # only save encoder
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output,
                            args.model + "_" + args.n + '_' + args.phase + '_' + str(args.ratio) + '.pt'))
            print("Saving model ", os.path.join(args.output, args.model + "_" + args.n + '_' + args.phase + '_' + str(
                args.ratio) + '.pt'))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Ea`rly Stopping")
                break
        sys.stdout.flush()
