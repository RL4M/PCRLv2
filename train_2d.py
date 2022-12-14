from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
from utils import adjust_learning_rate, AverageMeter
from models import PCRLv2

try:
    from apex import amp, optimizers
except ImportError:
    pass


# from koila import LazyTensor, lazy

def Normalize(x):
    norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
    x = x.div(norm_x)
    return x


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def mixup_data(x, alpha=1.0, index=None, lam=None, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = lam

    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cuda()
    else:
        index = index

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam, index


def train_pcrlv2(args, data_loader, out_channel=3):
    train_loader = data_loader['train']
    # create model and optimizer
    model = PCRLv2()

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = nn.DataParallel(model)

    criterion = nn.MSELoss().cuda()
    cosine = nn.CosineSimilarity().cuda()
    cudnn.benchmark = True
    loss_list = []
    mg_loss_list = []
    for epoch in range(0, args.epochs + 1):


        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        loss, mg_loss, prob = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine)
        loss_list.append(loss)
        mg_loss_list.append(mg_loss)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # save model
        if epoch % 100 == 0 or epoch == 240:
            # saving the model
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.model.encoder.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = os.path.join(args.output,
                                     args.model + "_" + args.n + '_' + args.phase + '_' + str(
                                         args.ratio) + '_' + str(epoch) + '.pt')
            torch.save(state, save_file)
            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def cos_loss(cosine, output1, output2):
    index = random.randint(0, len(output1) - 1)
    sample1 = output1[index]
    sample2 = output2[index]
    loss = -(cosine(sample1[1], sample2[0].detach()).mean() + cosine(sample2[1],
                                                                     sample1[0].detach()).mean()) * 0.5
    return loss, index


def train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine):
    """
    one epoch training for instance discrimination
    """

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    all_loss_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, gt, gt2, local_views) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = input1.size(0)
        x1 = input1.float().cuda()
        x2 = input2.float().cuda()
        gt = gt.float().cuda()
        decoder_outputs1, mask1, middle_masks1 = model(x1)
        decoder_outputs2, mask2, _ = model(x2)
        # print(len(local_views), local_views[0].shape)
        loss2, index2 = cos_loss(cosine, decoder_outputs1, decoder_outputs2)
        local_loss = 0.0
        local_input = torch.cat(local_views, dim=0)# 6 * bsz, 3, 96, 96
        local_views_outputs, _, _ = model(local_input, local=True)# 4 * 2 * [6 * bsz, 3, 96, 96]
        # print(len(local_views_outputs),local_views_outputs[0].shape)
        local_views_outputs = [torch.stack(t) for t in local_views_outputs]
       #  print(local_views_outputs[0].shape)
        for i in range(len(local_views)):
            # local_views_outputs, _, _ = model(local_views[i], local=True)
            local_views_outputs_tmp = [t[:, bsz * i: bsz * (i + 1)] for t in local_views_outputs]
            loss_local_1, _ = cos_loss(cosine, decoder_outputs1, local_views_outputs_tmp)
            loss_local_2, _ = cos_loss(cosine, decoder_outputs2, local_views_outputs_tmp)
            local_loss += loss_local_1
            local_loss += loss_local_2
        local_loss = local_loss / (2 * len(local_views))
        loss1 = criterion(mask1, gt)
        beta = 0.5 * (1. + math.cos(math.pi * epoch / 240))
        loss4 = beta * criterion(middle_masks1[index2], gt)
        loss = loss1 + loss2 + local_loss + loss4
        # ===================backward=====================
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # clip_value = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # ===================meters=====================
        mg_loss_meter.update(loss1.item(), bsz)
        loss_meter.update(loss2.item(), bsz)
        prob_meter.update(local_loss, bsz)
        all_loss_meter.update(loss.item(), bsz)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'cos_loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                  'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                  'local loss {prob.val:.3f} ({prob.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, c2l_loss=loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
            sys.stdout.flush()

    return loss_meter.avg, mg_loss_meter.avg, prob_meter.avg
