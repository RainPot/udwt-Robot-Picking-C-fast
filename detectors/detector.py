import torch.nn as nn
from backbones.resnet import Bottleneck
import torch.nn.functional as F
import torch


class CenterNetDetector(nn.Module):
    def __init__(self, planes, hm=True):
        super(CenterNetDetector, self).__init__()
        self.hm = hm
        self.detect_layer = nn.Sequential(
            BasicCov(3, 256, 256, with_bn=False),
            nn.Conv2d(256, planes, (1, 1))
        )
        if self.hm:
            self.detect_layer[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        output = self.detect_layer(x)
        return output


class BasicCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(BasicCov, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class FasterRCNNDetector(nn.Module):
    def __init__(self):
        super(FasterRCNNDetector, self).__init__()

        self.top_layer = Bottleneck(inplanes=256, planes=64)
        self.regressor = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, feat):
        feat = self.top_layer(feat)
        feat = F.adaptive_avg_pool2d(feat, 1)
        reg = self.regressor(feat)
        reg = reg.view(reg.size(0), reg.size(1))
        return reg


class CenterNetWHDetector(nn.Module):
    def __init__(self, planes, hm=True, num_stacks=2):
        super(CenterNetWHDetector, self).__init__()
        self.hm = hm
        self.num_stacks = num_stacks
        self.detect_conv_layer = nn.Sequential(
            BasicCov(3, 256, 256, with_bn=False),
        )

        self.detect_H_layer = nn.Sequential(
            HCov(17, 256, planes, with_bn=False)
        )

        self.detect_W_layer = nn.Sequential(
            WCov(17, 256, planes, with_bn=False)
        )

    def forward(self, input):
        conv = self.detect_conv_layer(input)
        H = self.detect_H_layer(conv)
        W = self.detect_W_layer(conv)
        H = H.view(H.size(0), -1, 1, H.size(2), H.size(3))
        W = W.view(W.size(0), -1, 1, W.size(2), W.size(3))
        output = torch.cat((W, H), dim=2).view(H.size(0), -1, H.size(3), H.size(4))
        return output


class HCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(HCov, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, 1), padding=(pad, 0), stride=(stride, stride), bias=not with_bn)

    def forward(self, x):
        conv = self.conv(x)
        return conv


class WCov(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(WCov, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (1, k), padding=(0, pad), stride=(stride, stride), bias=not with_bn)

    def forward(self, x):
        conv = self.conv(x)
        return conv


