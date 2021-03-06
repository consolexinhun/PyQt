"""Base Model for Semantic Segmentation"""
import torch.nn as nn

# import sys, os
# sys.path.append(os.path.abspath(".."))

from ..base_models.resnetv1b import resnet50_v1s
# from base_models.resnetv1b import resnet50_v1s

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, backbone='resnet50', pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

