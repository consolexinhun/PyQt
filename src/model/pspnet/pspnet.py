"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys, os
# sys.path.append(os.path.abspath(".."))

from ..common.segbase import SegBaseModel
# from common.segbase import SegBaseModel

__all__ = ['PSPNet', 'get_psp', 'get_psp_resnet50_voc', 'get_psp_resnet50_ade', 'get_psp_resnet101_voc',
           'get_psp_resnet101_ade', 'get_psp_resnet101_citys', 'get_psp_resnet101_coco']

from registry import MODEL

class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017
    """

    def __init__(self, nclass, backbone='resnet50', pretrained_base=True, **kwargs):
        super(PSPNet, self).__init__(nclass=nclass, backbone=backbone, pretrained_base=pretrained_base)
        self.head = _PSPHead(nclass, **kwargs)

        self.concat_global_local = kwargs.get("concat_global_local", False)
        if self.concat_global_local:
            self.pretrained.conv1 = nn.Sequential(
                nn.Conv2d(6, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )

    def forward(self, x):
        size = x.size()[2:]
        _, _, _, c4 = self.base_forward(x)
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
            pretrained_base=True, **kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    Examples
    --------
    >>> model = get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """

    num_classes = 1
    model = PSPNet(num_classes, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model


@MODEL.register("psp_resnet50_voc")
def get_psp_resnet50_voc(**kwargs):
    return get_psp(dataset='pascal_voc', backbone='resnet50', **kwargs)


def get_psp_resnet50_ade(**kwargs):
    return get_psp('ade20k', 'resnet50', **kwargs)


def get_psp_resnet101_voc(**kwargs):
    return get_psp('pascal_voc', 'resnet101', **kwargs)


def get_psp_resnet101_ade(**kwargs):
    return get_psp('ade20k', 'resnet101', **kwargs)


def get_psp_resnet101_citys(**kwargs):
    return get_psp('citys', 'resnet101', **kwargs)


def get_psp_resnet101_coco(**kwargs):
    return get_psp('coco', 'resnet101', **kwargs)


if __name__ == '__main__':
    model = get_psp_resnet50_voc()
    print(model)
    # summary(model, input_size=(3, 512, 512), device="cpu")
    # img = torch.randn(4, 3, 512, 512)
    # output = model(img)
    # print(output.shape)

