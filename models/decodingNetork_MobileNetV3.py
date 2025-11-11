import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as c

class H_Swish(nn.Module):
    def __init__(self, inplace=True):
        super(H_Swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3.) / 6.

class H_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(H_Sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.) / 6.

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=nn.ReLU):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se=False, activation=H_Swish):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.se = se
        self.activation = activation

        hidden_channels = in_channels * expand_ratio
        self.conv1 = ConvBNRelu(in_channels, hidden_channels, kernel_size=1, activation=self.activation)
        self.conv2 = ConvBNRelu(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, groups=hidden_channels, activation=self.activation)
        if self.se:
            self.se_module = SqueezeExcitation(hidden_channels)
        self.conv3 = ConvBNRelu(hidden_channels, out_channels, kernel_size=1, activation=nn.Identity)

        if stride == 1 and in_channels != out_channels:
            self.shortcut = ConvBNRelu(in_channels, out_channels, kernel_size=1, activation=nn.Identity)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se:
            out = self.se_module(out)
        out = self.conv3(out)
        if self.shortcut:
            out = out + self.shortcut(x)
        return out

class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(MobileNetV3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_c = base_c

        self.layers = nn.Sequential(
            ConvBNRelu(in_channels, base_c, activation=H_Swish),
            MobileNetV3Block(base_c, base_c * 2, kernel_size=3, stride=2, expand_ratio=1, se=True, activation=H_Swish),
            MobileNetV3Block(base_c * 2, base_c * 4, kernel_size=3, stride=2, expand_ratio=1, se=True, activation=H_Swish),
            MobileNetV3Block(base_c * 4, base_c * 8, kernel_size=3, stride=2, expand_ratio=1, se=True, activation=H_Swish),
            MobileNetV3Block(base_c * 8, base_c * 16, kernel_size=3, stride=2, expand_ratio=1, se=True, activation=H_Swish),
            nn.Conv2d(base_c * 16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)

class decodingNetwork(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(decodingNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.PixelUnshuffle(c.psf),
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            MobileNetV3(in_channels=64, out_channels=64),  # 使用 MobileNetV3 替代 DenseNet
            nn.Conv2d(64, 64, kernel_size=3, stride=down_ratio_l2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            MobileNetV3(in_channels=64, out_channels=64),  # 使用第二个 MobileNetV3
            nn.Conv2d(64, output_channels, kernel_size=3, stride=down_ratio_l3, padding=1),
            nn.Sigmoid(),
            nn.PixelShuffle(c.psf),
            nn.Upsample(size=(c.secret_image_size, c.secret_image_size), mode='bilinear', align_corners=False)
        )

    def forward(self, input):
        return self.layers(input)

class dec_img(nn.Module):
    def __init__(self, input_channel=3, output_channels=3):
        super(dec_img, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            MobileNetV3(in_channels=64, out_channels=64),  # 使用 MobileNetV3 替代 DenseNet
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            MobileNetV3(in_channels=64, out_channels=64),  # 使用第二个 MobileNetV3
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)

def init_weights(model, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)