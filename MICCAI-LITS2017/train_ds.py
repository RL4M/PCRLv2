"""

训练脚本
"""

import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para
import argparse

try:
    from apex import amp, optimizers
except ImportError:
    pass

# 设置visdom
parser = argparse.ArgumentParser(description='Self Training benchmark')
parser.add_argument('--ratio', default=1.0, type=float)
parser.add_argument('--weight', default=None, type=str)
parser.add_argument('--out', default='./weight', type=str)
parser.add_argument('--gpus', default='0,1,2,3', type=str)
args = parser.parse_args()
save_path = os.path.join(args.out, str(args.ratio))
if not os.path.exists(save_path):
    os.mkdir(save_path)
step_list = [0]

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
cudnn.benchmark = para.cudnn_benchmark
print('loading weight')
weight_path = args.weight
if weight_path is not None:
    encoder_dict = net.state_dict()
    checkpoint = torch.load(weight_path)
    state_dict = checkpoint['state_dict']
    pretrain_dict = {k: v for k, v in state_dict.items() if
                     k in encoder_dict and 'down' in k and 'down_tr64' not in k}
    print(pretrain_dict.keys())
    encoder_dict.update(pretrain_dict)
    net.load_state_dict(encoder_dict)
net = torch.nn.DataParallel(net)
net = net.cuda()
# 定义Dateset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'),
                   training=True, ratio=args.ratio)
valid_ds = Dataset(os.path.join(para.valid_set_path, 'ct'), os.path.join(para.valid_set_path, 'seg'), training=False)
# 定义数据加载
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
val_dl = DataLoader(valid_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
# 挑选损失函数
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]
# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# 深度监督衰减系数
alpha = para.alpha
best_loss = 1000.
# 训练网络
start = time()
for epoch in range(para.Epoch):
    net.train()
    lr_decay.step()

    mean_loss = []
    mean_valid_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        # loss5 = loss_func(outputs[3], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4
        # #
        mean_loss.append(loss4.item())
        # loss1 = loss_func(outputs[0], seg)
        # loss2 = loss_func(outputs[1], seg)
        # loss3 = loss_func(outputs[2], seg)
        # loss4 = loss_func(outputs[3], seg)

        # loss = (loss1 + loss2) * alpha + loss3
        #
        # mean_loss.append(loss3.item())

        opt.zero_grad()
        loss.backward()

        # with amp.scale_loss(loss, opt) as scaled_loss:
        #     scaled_loss.backward()

        opt.step()

        if step % 5 is 0:
            step_list.append(step_list[-1] + 1)
            # viz.line(X=np.array([step_list[-1]]), Y=np.array([loss4.item()]), win=win, update='append')

            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)
    net.eval()
    with torch.no_grad():
        for step, (ct, seg) in enumerate(val_dl):
            ct = ct.cuda()
            seg = seg.cuda()

            outputs = net(ct)

            # loss1 = loss_func(outputs[0], seg)
            # loss2 = loss_func(outputs[1], seg)
            # loss3 = loss_func(outputs[2], seg)
            # loss5 = loss_func(outputs[3], seg)
            loss4 = loss_func(outputs[3], seg)

            # loss = (loss1 + loss2 + loss3) * alpha + loss4
            # #
            mean_valid_loss.append(loss4.item())
        valid_loss = sum(mean_valid_loss) / len(mean_valid_loss)
        print('valid loss is:', valid_loss)
    if valid_loss < best_loss:
        print('saving model')
        best_loss = valid_loss
        torch.save(net.state_dict(), os.path.join(save_path, 'net.pth'))
    # # 保存模型
    # if epoch % 50 is 0 and epoch is not 0:
    #     # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
    #     torch.save(net.state_dict(), './module3/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    # 对深度监督系数进行衰减
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
