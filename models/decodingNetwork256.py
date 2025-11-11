import torch
import torch.nn as nn
import config as c

class decodingNetwork(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(decodingNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.PixelUnshuffle(c.psf),  # 将像素重排 (相当于下采样)
            nn.Conv2d(input_channel, 104, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(104),
            # nn.Conv2d(96, 96, kernel_size=3, stride=down_ratio_l2, padding=1),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.InstanceNorm2d(96),
            nn.Conv2d(104, 104, kernel_size=3, stride=down_ratio_l2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(104),
            nn.Conv2d(104, output_channels, kernel_size=3, stride=down_ratio_l3, padding=1),
            nn.Sigmoid(),
            nn.PixelShuffle(c.psf),  # 将通道重排 (上采样)
            # 以双线性插值强制输出到 128x128
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.layers(x)


class dec_img(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(dec_img, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 104, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(104),
            # nn.Conv2d(96, 96, kernel_size=3, stride=down_ratio_l2, padding=1),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.InstanceNorm2d(96),
            nn.Conv2d(104, 104, kernel_size=3, stride=down_ratio_l2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(104),
            nn.Conv2d(104, output_channels, kernel_size=3, stride=down_ratio_l3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.layers(x)


class dec_img_new(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(dec_img_new, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 104, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm2d(104),
            # nn.Conv2d(96, 96, kernel_size=3, stride=down_ratio_l2, padding=1),
            # nn.SiLU(inplace=True),
            # nn.InstanceNorm2d(96),
            nn.Conv2d(104, 104, kernel_size=3, stride=down_ratio_l2, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm2d(104),
            nn.Conv2d(104, output_channels, kernel_size=3, stride=down_ratio_l3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.layers(x)


class stegan_dec(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(stegan_dec, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, output_channels, kernel_size=3, stride=down_ratio_l3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.layers(x)


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
