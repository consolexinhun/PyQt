from torch import nn
import torch
from torch.nn import functional as F

from registry import LOSS

@LOSS.register("DiceLossV1")
class DiceLossV1(nn.Module):
    """
    different from normal dice loss
    for image with no positive mask, only calculate dice for background
    for positive image, only calculate positive pixel
    """
    def __init__(self, smooth=1e-7, weight=None):
        """
        Diceloss for segmentation
        :param smooth: smooth value for fraction
        :param weight: class weight
        """
        super(DiceLossV1, self).__init__()
        self.smooth = smooth
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        code from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        Note that PyTorch optimizers minimize a loss. In this case,
        we would like to maximize the dice loss so we return the negated dice loss.
        :param gt: a tensor of shape [B, 1 , H, W]
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logit of the model.
        :param reduction:
        :return:
            dice_loss: dice loss
        """
        num_classes = logits.shape[1]
        gt = gt.long()
        if num_classes == 1:
            gt_1_hot = torch.eye(num_classes + 1)[gt.squeeze(1)]  # B,H,W,2
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,2,H,W

            pos_prob = torch.sigmoid(logits)  # B,1,H,W
            neg_prob = 1 - pos_prob  # B,1,H,W
            probas = torch.cat([neg_prob, pos_prob], dim=1)  # B,2,H,W

        gt_1_hot = gt_1_hot.type(logits.type())

        # whether gt have pos pixel
        is_pos = gt.gt(0).sum(dim=(1, 2, 3)).gt(0)  # B 哪些样本是大于0的，返回 bool 类型

        dims = tuple(range(2, logits.ndimension()))  # H,W
        intersection = torch.sum(probas * gt_1_hot, dims)  # B,cls
        cardinality = torch.sum(probas + gt_1_hot, dims)  # B,cls
        dice_coff = 2 * intersection / (cardinality + self.smooth)  # B,cls
        
        dice_loss = 1 - dice_coff
        if self.weight is not None:
            weight = self.weight.to(logits.device)  # cls
            dice_loss = weight * dice_loss  # B, cls

        dice_loss = (torch.where(is_pos, dice_loss[:, 1:].mean(1), dice_loss[:, 0])).mean()
        # dice_loss[:, 1:].mean(1) 每个样本阳性类别的平均dice
        # torch.where(condition, x, y) 如果 condition True，选 x，否则选 y

        return dice_loss