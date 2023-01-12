"""

网络定义脚本
"""

import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

import parameter as para


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_chan, eps=1e-05, affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        else:
            raise ValueError('normalization type {} is not supported'.format(norm))

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('activation type {} is not supported'.format(act))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, norm, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act, norm)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act, norm)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act, norm)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act, norm)

    return nn.Sequential(layer1, layer2)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act, norm):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, norm, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        return self.ops(concat)


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act, norm):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act, norm)

    def forward(self, x):
        return self.ops(x)


class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='gn', in_channels=1, training=True):
        super(UNet3D, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        # self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2)
        self.training = training
        self.down_tr64 = DownTransition(in_channels, 0, act, norm)
        self.down_tr128 = DownTransition(32, 1, act, norm)
        self.down_tr256 = DownTransition(64, 2, act, norm)
        self.down_tr512 = DownTransition(128, 3, act, norm)
        self.up_tr256 = UpTransition(256, 256, 2, act, norm)
        self.up_tr128 = UpTransition(128, 128, 1, act, norm)
        self.up_tr64 = UpTransition(64, 64, 0, act, norm)
        self.out_tr = OutputTransition(32, n_class)
        # self.up4 = OutputTransition(64, 32)
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_out64 = self.down_tr64(x)
        skip_out64 = F.dropout(skip_out64, para.drop_rate, self.training)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out128 = F.dropout(skip_out128, para.drop_rate, self.training)
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        skip_out256 = F.dropout(skip_out256, para.drop_rate, self.training)
        out512 = self.down_tr512(self.maxpool(skip_out256))
        out512 = F.dropout(out512, para.drop_rate, self.training)
        # output1 = self.map1(out512)
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_256 = F.dropout(out_up_256, para.drop_rate, self.training)
        output2 = self.map2(out_up_256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_128 = F.dropout(out_up_128, para.drop_rate, self.training)
        output3 = self.map3(out_up_128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out_up_64 = F.dropout(out_up_64, para.drop_rate, self.training)
        # output3 = self.map3(out_up_64)
        # out = self.out_tr(out_up_64)
        # out = self.up4(out_up_64)
        out = self.map4(out_up_64)
        # print(out.shape, output2.shape, output3.shape, output1.shape)
        # if training:
        #     return output2, output3, out
        # return out
        return output2, output3, out


class UNet3D2(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='gn', in_channels=1, training=True):
        super(UNet3D2, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        # self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2)
        self.training = training
        self.down_tr64 = DownTransition(32, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.up_tr256 = UpTransition(512, 512, 2, act, norm)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm)
        # self.up_tr32 = UpTransition(64, 64, -1, act, norm)
        self.out_tr = OutputTransition(32, n_class)
        self.upsample1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.relu = nn.ReLU()
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU()
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU()
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU()
        )
        # self.up4 = OutputTransition(64, 32)
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )
        self.map5 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_out32 = self.conv1(x)
        # skip_out32 = self.maxpool(skip_out32)
        skip_out32 = F.dropout(skip_out32, para.drop_rate, self.training)
        skip_out64 = self.down_tr64(self.maxpool(skip_out32))
        skip_out64 = F.dropout(skip_out64, para.drop_rate, self.training)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out128 = F.dropout(skip_out128, para.drop_rate, self.training)
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        skip_out256 = F.dropout(skip_out256, para.drop_rate, self.training)
        out512 = self.down_tr512(self.maxpool(skip_out256))
        out512 = F.dropout(out512, para.drop_rate, self.training)
        # output1 = self.map1(out512)
        # print(out512.shape, skip_out256.shape)
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_256 = F.dropout(out_up_256, para.drop_rate, self.training)
        output2 = self.map2(out_up_256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_128 = F.dropout(out_up_128, para.drop_rate, self.training)
        output3 = self.map3(out_up_128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out_up_64 = F.dropout(out_up_64, para.drop_rate, self.training)
        # output5 = self.map5(out_up_64)
        # print(skip_out64.shape, skip_out32.shape, out_up_64.shape)
        out_up_32 = self.up_conv1(out_up_64)
        out_up_32 = torch.cat((skip_out32, out_up_32), dim=1)
        out_up_32 = F.dropout(out_up_32, para.drop_rate, self.training)
        output4 = self.map1(out_up_32)
        out_up_32 = self.upsample1(out_up_32)
        out_up_32 = self.relu(out_up_32)
        # output3 = self.map3(out_up_64)
        # out = self.out_tr(out_up_64)
        # out = self.up4(out_up_64)
        out = self.map4(out_up_32)
        # print(out.shape, output3.shape, output2.shape)
        # print(out.shape, output2.shape, output3.shape, output1.shape)
        # if training:
        #     return output2, output3, out
        # return out
        return output2, output3, output4, out


class UNet3D3(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='gn', in_channels=1, training=True):
        super(UNet3D3, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        # self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2)
        self.training = training
        self.down_tr64 = DownTransition(32, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.up_tr256 = UpTransition(512, 512, 2, act, norm)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm)
        # self.up_tr32 = UpTransition(64, 64, -1, act, norm)
        self.out_tr = OutputTransition(32, n_class)
        self.upsample1 = nn.Sequential(nn.ConvTranspose3d(32, 32, 2, 2),
                                       nn.ReLU())
        self.conv2 = nn.Conv3d(64 + 32, 32, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU(inplace=True)
        )
        # self.up4 = OutputTransition(64, 32)
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )
        self.map5 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_out32 = self.conv1(x)
        # skip_out32 = self.maxpool(skip_out32)
        skip_out32 = F.dropout(skip_out32, para.drop_rate, self.training)
        skip_out64 = self.down_tr64(self.maxpool(skip_out32))
        skip_out64 = F.dropout(skip_out64, para.drop_rate, self.training)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out128 = F.dropout(skip_out128, para.drop_rate, self.training)
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        skip_out256 = F.dropout(skip_out256, para.drop_rate, self.training)
        out512 = self.down_tr512(self.maxpool(skip_out256))
        out512 = F.dropout(out512, para.drop_rate, self.training)
        # output1 = self.map1(out512)
        # print(out512.shape, skip_out256.shape)
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_256 = F.dropout(out_up_256, para.drop_rate, self.training)
        output2 = self.map2(out_up_256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_128 = F.dropout(out_up_128, para.drop_rate, self.training)
        output3 = self.map3(out_up_128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out_up_64 = F.dropout(out_up_64, para.drop_rate, self.training)
        # output5 = self.map5(out_up_64)
        # print(skip_out64.shape, skip_out32.shape, out_up_64.shape)
        out_up_32 = self.up_conv1(out_up_64)
        out_up_32 = torch.cat((skip_out32, out_up_32), dim=1)
        out_up_32 = self.conv2(out_up_32)
        out_up_32 = F.dropout(out_up_32, para.drop_rate, self.training)
        output4 = self.map1(out_up_32)
        out_up_32 = self.upsample1(out_up_32)
        # out_up_32 = self.relu(out_up_32)
        # output3 = self.map3(out_up_64)
        # out = self.out_tr(out_up_64)
        # out = self.up4(out_up_64)
        out = self.map4(out_up_32)
        # print(out.shape, output3.shape, output2.shape)
        # print(out.shape, output2.shape, output3.shape, output1.shape)
        # if training:
        #     return output2, output3, out
        # return out
        return output2, output3, output4, out

class UNet3D4(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='gn', in_channels=1, training=True):
        super(UNet3D4, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2)
        self.training = training
        self.down_tr64 = DownTransition(32, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.up_tr256 = UpTransition(512, 512, 2, act, norm)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm)
        # self.up_tr32 = UpTransition(64, 64, -1, act, norm)
        self.out_tr = OutputTransition(32, n_class)
        self.upsample1 = nn.Sequential(nn.ConvTranspose3d(32, 32, 2, 2),
                                       nn.ReLU())
        self.conv2 = nn.Conv3d(64 + 32, 32, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.ReLU(inplace=True)
        )
        # self.up4 = OutputTransition(64, 32)
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )
        self.map5 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_out32 = self.conv1(x)
        # down_out32 = self.maxpool(skip_out32)
        skip_out32 = F.dropout(skip_out32, para.drop_rate, self.training)
        skip_out64 = self.down_tr64(self.maxpool(skip_out32))
        skip_out64 = F.dropout(skip_out64, para.drop_rate, self.training)
        skip_out128 = self.down_tr128(self.maxpool(skip_out64))
        skip_out128 = F.dropout(skip_out128, para.drop_rate, self.training)
        skip_out256 = self.down_tr256(self.maxpool(skip_out128))
        skip_out256 = F.dropout(skip_out256, para.drop_rate, self.training)
        out512 = self.down_tr512(self.maxpool(skip_out256))
        out512 = F.dropout(out512, para.drop_rate, self.training)
        # output1 = self.map1(out512)
        # print(out512.shape, skip_out256.shape)
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_256 = F.dropout(out_up_256, para.drop_rate, self.training)
        output2 = self.map2(out_up_256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_128 = F.dropout(out_up_128, para.drop_rate, self.training)
        output3 = self.map3(out_up_128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out_up_64 = F.dropout(out_up_64, para.drop_rate, self.training)
        output4 = self.map5(out_up_64)
        # print(skip_out64.shape, skip_out32.shape, out_up_64.shape)
        out_up_32 = self.up_conv1(out_up_64)
        out_up_32 = torch.cat((skip_out32, out_up_32), dim=1)
        out_up_32 = self.conv2(out_up_32)
        # out_up_32 = F.dropout(out_up_32, para.drop_rate, self.training)
        # output4 = self.map1(out_up_32)
        # out_up_32 = self.upsample1(out_up_32)
        # out_up_32 = self.relu(out_up_32)
        # output3 = self.map3(out_up_64)
        # out = self.out_tr(out_up_64)
        # out = self.up4(out_up_64)
        out = self.map4(out_up_32)
        # print(out.shape, output3.shape, output2.shape)
        # print(out.shape, output2.shape, output3.shape, output1.shape)
        # if training:
        #     return output2, output3, out
        # return out
        return output2, output3, output4, out

class ResUNet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """

    def __init__(self, training):
        super().__init__()

        self.training = training

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, para.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, para.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, para.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, para.drop_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)



# net = ResUNet(training=True)
net = UNet3D3(training=True)
net.apply(init)

print('net total parameters:', sum(param.numel() for param in net.parameters()))
