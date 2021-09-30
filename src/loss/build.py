from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
import torch

from .loss import DiceLossV1
from registry import LOSS

def build_loss(cfg):
    name = cfg.MODEL.LOSS
    weight = cfg.MODEL.LOSS_WEIGHT

    seg_loss = LOSS[name](weight=weight)
    return seg_loss

    # cls_loss = BCEWithLogitsLoss()
    # return LossWrapper(seg_loss, cls_loss)


# class LossWrapper(nn.Module):
#     """
#     wrapper of loss, because
#     1. model output is multilayer in deep supervision, need to calculate loss for each layer
#     2. mirror padding in transformation, need to remove padding area before calculate loss
#     """
#     def __init__(self, seg_loss, cls_loss, cls_weight=1):
#         super(LossWrapper, self).__init__()
#         self.seg_loss = seg_loss
#         self.cls_loss = cls_loss
#         self.cls_weight = cls_weight

#     def forward(self, gt, logits):
#         """
#         :param gt: tensor of shape B, C, H, W
#         :param logits: tensor of shape B, C, H, W
#         :return:
#         """

#         if isinstance(logits, list):
#             if len(logits) > 2:  # 训练的 loss
#                 # deep supervision with multiple output
#                 loss_result = logits[0].new_zeros(1)
#                 count = 0.
#                 for logit in logits:
#                     loss_result += self.seg_loss(gt, logit)
#                     count += 1
#                 loss_result /= count
#             else:
#                 # no deep supervision, or fusion deep supervision
#                 logits = logits[0]
#                 loss_result = self.seg_loss(gt, logits)

#         else:
#             # 验证和测试的 loss
#             # inference loss
#             loss_result = self.seg_loss(gt, logits)
#         return loss_result

