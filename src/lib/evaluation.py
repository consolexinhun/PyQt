from collections import defaultdict
from time import time

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import torch

def get_cls_label(pred_mask, th_ratio=0.003, th_pixel=30000):
    """
    func:
        获取类别标签，计算占整个区域的占比，超过0.3%并且大于3万个像素 为阳性，否则阴性
    args:
        pred_mask:(1, H, W) uint8, 0/1
    """
    pred_area = pred_mask.sum()
    total_area = pred_mask.shape[-1] * pred_mask.shape[-2]
    ratio = float(pred_area) / total_area
    return int(ratio >= th_ratio and pred_area >= th_pixel)


def dice_score(gt_mask, pred_mask, smooth=1e-7):
    """
    func:  
        calculate dice score of prediction
    
    :param gt_mask:  ndarray of shape [H, W]
    :param pred_mask:  ndarray of shape [C, H, W]
    :param smooth: smooth value
    :return: dice score of each class, average dice
    """
    gt_mask = gt_mask.to(torch.uint8)
    device = gt_mask.device

    num_classes = pred_mask.shape[0]
    if num_classes == 1:
        num_classes = 2
        pred_1_hot = torch.eye(2, dtype=torch.uint8)[pred_mask.squeeze(0).long()].to(device)  # H,W,2
    else:
        pred_mask = torch.argmax(pred_mask, dim=0).to(torch.uint8)  # H,W
        pred_1_hot = torch.eye(num_classes, dtype=torch.uint8)[pred_mask.long()].to(device)  # H,W,C

    gt_1_hot = torch.eye(num_classes, dtype=torch.uint8)[gt_mask.long()].to(device)  # H,W,C

    intersection = torch.sum(gt_1_hot * pred_1_hot, dim=(0, 1)).float()  # cls
    cardinality = torch.sum(gt_1_hot + pred_1_hot, dim=(0, 1)).float()  # cls
    dice_coff = 2. * intersection / (cardinality + smooth)  # cls

    # 返回每一类别的 dice ，和平均 dice，不过平均 dice 没有用到
    return dice_coff, torch.mean(dice_coff)



def do_evaluation(gt_masks, pred_masks, detail_info=False):
    """
    func:
        evaluate on model result, calculate dice,
    params:
        gt_masks: list of ndarray of shape [1, H, W](train) [H, W](val)
        pred_masks: list of ndarray of shape [1, H, W]
        detail_info: bool, whether return each image's metric result
    return:

    """
    # result_list = defaultdict(list)  # 创建一个字典，字典的值默认是 空列表[]
    result_list = {
        "gt_label": [],  # 原图的图像级别标签
        "pred_label": [],  # 预测的图像级别标签
        "dice_1": []       # 每个样本的 dice
    }

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        # gt_mask: train(1, H, W) val(H, W)     
        # pred_mask: (1, H, W)
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)
        # gt_mask: (H, W)

        pred_mask[pred_mask >= 0.5] = 1
        pred_mask[pred_mask < 0.5] = 0
        pred_mask.astype(np.uint8)

        ### 判断类别的逻辑 classification
        # 真实标签：只要有1就是阳性     预测标签：大于0.3%并且大于阈值为阳性
        gt_label = int(gt_mask.sum() > 0)
        pred_label = get_cls_label(pred_mask)
        result_list['gt_label'].append(gt_label)
        result_list['pred_label'].append(pred_label)

        # segmentation
        if gt_label == 1:
            # positive sample, calculate dice ，只计算 癌变类别的dice
            gt_mask = torch.from_numpy(gt_mask)
            pred_mask = torch.from_numpy(pred_mask)

            dices, dice_avg = dice_score(gt_mask, pred_mask)

            result_list['dice_1'].append(float(dices[1].cpu()))
        else:
            result_list['dice_1'].append(-1.0)

    dice_1 = [x for x in result_list['dice_1'] if x >= 0]  # 只计算阳性类别的 Dice
    mean_dice = np.mean(dice_1) if len(dice_1) > 0 else 0  # 所有样本 的 Dice 的平均

    try:
        auc = roc_auc_score(result_list['gt_label'], result_list['pred_label'])
    except ValueError:
        auc = -1
    acc = accuracy_score(result_list['gt_label'], result_list['pred_label'])
    precision = precision_score(result_list['gt_label'], result_list['pred_label'], zero_division=0)
    recall = recall_score(result_list['gt_label'], result_list['pred_label'], zero_division=0)

    result = {
        'dice': mean_dice,
        'auc': auc,
        'acc': acc,
        'precision': precision,
        'recall': recall
    }

    if detail_info:
        # result list 中包含每个patch的阳性dice_1，以及原图和预测的图像级标签 gt_label pred_label
        return result, result_list
    else:
        return result


def do_test_evaluation(gt_mask, pred_mask):
    """
    func:
        evaluate on model result, calculate dice,
    params:
        gt_masks: [H, W]
        pred_masks: [H, W]
    return:
        gt_label, pred_label, dice_1 都是标量
    """
    pred_mask[pred_mask >= 0.5] = 1
    pred_mask[pred_mask < 0.5] = 0
    pred_mask.astype(np.uint8)

    ### 判断类别的逻辑 classification
    # 真实标签：只要有1就是阳性     预测标签：大于0.3%并且大于阈值为阳性
    gt_label = int(gt_mask.sum() > 0)
    pred_label = get_cls_label(pred_mask)

    # segmentation
    if gt_label == 1:
        # positive sample, calculate dice ，只计算 癌变类别的dice
        gt_mask = torch.from_numpy(gt_mask)
        pred_mask = torch.from_numpy(pred_mask)

        dices, dice_avg = dice_score(gt_mask, pred_mask)
        dice_1 = float(dices[1].cpu())
    else:
        dice_1 = -1
    return gt_label, pred_label, dice_1



