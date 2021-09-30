import torch.nn as nn
import torch

# from .components import DeepSupervisionBlockV2

class UNet(nn.Module):
    def __init__(self, backbone, deep_sup='', ds_se=False, fusion_ds='', layer_nums=4, cls_branch=False):
        super(UNet, self).__init__()

        self.inc = backbone.inc
        self.encoder1 = backbone.down1
        self.encoder2 = backbone.down2
        self.encoder3 = backbone.down3
        self.encoder4 = backbone.down4

        self.decoder1 = backbone.up1
        self.decoder2 = backbone.up2
        self.decoder3 = backbone.up3
        self.decoder4 = backbone.up4
        self.out = backbone.outc

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        
        y4 = self.decoder4(x5, x4)
        y3 = self.decoder3(y4, x3)
        y2 = self.decoder2(y3, x2)
        y1 = self.decoder1(y2, x1)
        out_final = self.out(y1)

        if self.training:
            return out_final
        return torch.sigmoid(out_final)

if __name__ == '__main__':
    from vgg_unet import UNet16Layer4
    from torchvision.models import vgg16

    backbone = UNet16Layer4(backbone=vgg16(pretrained=True), n_classes=1)
    model = UNet(backbone, deep_sup="")

    a = torch.randn(2, 3, 512, 512)
    out = model(a)
    print(out.shape)

