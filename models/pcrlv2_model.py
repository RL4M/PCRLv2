import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md

from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        # self.projection_head = nn.Sequential( nn.Linear(out_channels, 2 * out_channels),
        #                                      nn.BatchNorm1d(2 * out_channels),
        #                                      nn.ReLU(inplace=True),  # first layer
        #                                      nn.Linear(2 * out_channels, 2 * out_channels, bias=False),
        #                                      nn.BatchNorm1d(2 * out_channels),
        #                                      nn.ReLU(inplace=True),
        #                                      nn.Linear(2 * out_channels, out_channels),
        #                                      nn.BatchNorm1d(out_channels))
        self.bn = nn.BatchNorm1d(out_channels)
        self.deep_supervision_head = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                                   nn.BatchNorm2d(out_channels),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv2d(out_channels, 3, kernel_size=1))
        # self.bn.bias.requires_grad = False
        self.predictor_head = nn.Sequential(nn.Linear(out_channels, 2 * out_channels),
                                            nn.BatchNorm1d(2 * out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2 * out_channels, out_channels))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # if skip is not None:
        #     x = torch.cat([x, skip], dim=1)
        #     x = self.attention1(x)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        b = x.shape[0]
        x_mask = self.deep_supervision_head(x)
        # x_pro = self.projection_head(F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1))
        x_pro = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1)
        x_pro = self.bn(x_pro)
        x_pre = self.predictor_head(x_pro)
        return x, x_pro, x_pre, x_mask


class PCRLv2Decoder(nn.Module):
    def __init__(
            self,
            # decoder,
            encoder_channels=512,
            n_class=3,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None

    ):
        super().__init__()
        # self.decoder = decoder
        # self.segmentation_head = segmentation_head
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        # self.conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1)
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        # self.drop = nn.Dropout2d(0.5)
        initialize_decoder(self.blocks)

    def forward(self, features, local=False):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        decoder_outs = []
        middle_masks = []
        for i, decoder_block in enumerate(self.blocks):
            b = x.shape[0]
            skip = skips[i] if i < len(skips) else None
            x, x_pro, x_pre, x_mask = decoder_block(x, skip)
            decoder_outs.append((x_pro, x_pre))
            if not local:
                middle_out = F.interpolate(x_mask, scale_factor=2 ** (4 - i), mode='bilinear')
                middle_masks.append(middle_out)
        return decoder_outs, x, middle_masks


class PCRLv2(nn.Module):
    def __init__(self, n_class=3, low_dim=128):
        super(PCRLv2, self).__init__()
        self.model = smp.Unet('resnet18', in_channels=3, classes=n_class)
        self.model.decoder = PCRLv2Decoder(self.model.encoder.out_channels)

    def forward(self, x, local=False):
        features = self.model.encoder(x)
        decoder_outputs, x, middle_masks = self.model.decoder(features)
        masks = None
        if not local:
            masks = self.model.segmentation_head(x)
        return decoder_outputs, masks, middle_masks
