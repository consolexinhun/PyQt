

from registry import MODEL

from .unet.vgg_unet import unet16layer4
from .pspnet.pspnet import get_psp_resnet50_voc
from .fcn.fcn import get_fcn32s_vgg16_voc
from .unet_family.resunet import R2U_Net 
from .deeplab.deeplabv3plus import DeepLabV3Plus


def build_model(cfg, pre_train=True):
    name = cfg.MODEL.MODEL
    model = MODEL[name](
        pre_train=pre_train,
        n_classes=cfg.DATA.NUM_CLS,
        deep_sup=cfg.MODEL.DEEP_SUP
    )
    return model

def concat_build_model(cfg, pre_train=True):
    model = get_psp_resnet50_voc(
        pre_train=pre_train,
        n_classes=cfg.DATA.NUM_CLS,
        deep_sup=cfg.MODEL.DEEP_SUP,
        concat_global_local=True
    )
    return model