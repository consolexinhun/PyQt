
from torchvision.models import vgg16
import torch.nn as nn


from registry import MODEL
from .components import TernausUp
from .unet_frame import UNet

@MODEL.register("UNet16Layer4")
def unet16layer4(**kwargs):
    pre_train=kwargs.get("pre_train", True)
    num_classes = kwargs.get('n_classes', 1)

    vgg_model = vgg16(pretrained=pre_train)
    backbone = UNet16Layer4(backbone=vgg_model, n_classes=num_classes)

    deep_sup = kwargs.get("deep_sup", "")
    return UNet(backbone, deep_sup)


class UNet16Layer4:
    def __init__(self, backbone, n_classes=1, num_filters=32):

        super().__init__()
        if n_classes == 2:
            n_classes=1

        self.n_classes = n_classes
        self.encoder = backbone.features

        self.inc = self.encoder[0:4]
        self.down1 = self.encoder[4:9]
        self.down2 = self.encoder[9:16]
        self.down3 = self.encoder[16:23]
        self.down4 = self.encoder[23:30]

        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512]

        
