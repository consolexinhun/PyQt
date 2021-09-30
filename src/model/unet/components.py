import torch.nn as nn
import torch


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBNRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Sequential(conv3x3(in_, out),
                                  nn.BatchNorm2d(out))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class TernausUp(nn.Module):
    """
    TernausNet upsample model
    """
    def __init__(self, in_channels, in_up_channels, skip_conn_channels, out_channels, bn=False):
        super().__init__()
        if bn:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_up_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_up_channels),
                nn.ReLU(inplace=True))
            self.conv = ConvBNRelu(in_up_channels + skip_conn_channels, out_channels)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_up_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True))
            self.conv = ConvRelu(in_up_channels + skip_conn_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        y = self.conv(torch.cat([x1, x2], 1))

        return y



class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # b, c, h, w
        y = self.squeeze(x).view(b, c)  # b, c
        weight = self.excitation(y).view(b, c, 1, 1)  # b, c, 1, 1
        return x * weight.expand_as(x)



class ResBlock(nn.Module):
    def __init__(self, in_ch, inter_ch=64, se_module=False):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        self.se_module = se_module
        if se_module:
            self.se = SELayer(in_ch)

    def forward(self, x):
        identity = x

        x = self.conv(x)
        if self.se_module:
            x = self.se(x)
        x += identity
        y = self.relu(x)
        return y

class DeepSupervisionBlockV2(nn.Module):
    def __init__(self, in_ch, num_cls, up_scale, se_module=False):
        super(DeepSupervisionBlockV2, self).__init__()
        self.up = nn.Sequential(
            ResBlock(in_ch, se_module=se_module),
            nn.Conv2d(in_ch, num_cls, kernel_size=(1, 1)),
            nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True),
            nn.Conv2d(num_cls, num_cls, kernel_size=1)
        )
        initialize_weights(self.up)

    def forward(self, x):
        x = self.up(x)
        return x

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)