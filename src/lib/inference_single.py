
import torch
import numpy as np

from .post_process import resize_ndarry, merge_patch

def inference_single_dense_crop(dataloader, image, index, model, device, cfg):
    """
    func:
        密集裁剪
    return: 
        pred_masks_binary
    """
    ori_img =  dataloader.dataset.get_img(index[0])
    pred_masks_binary = []
    # 每次送 8 个补丁进去
    # test_batch_size = cfg.SOLVER.BATCH_SIZE_PER_IMG_TEST
    test_batch_size = 4
    patch_nums, _, h, w = image.shape
    pred_patches_mask = torch.zeros(patch_nums, h, w).float()

    for i in range(0, patch_nums, test_batch_size):
        start_idx = i
        end_idx = min(i + test_batch_size, patch_nums)
        test_patch_batch = image[start_idx: end_idx].to(device)
        test_patch_batch_mask = model(test_patch_batch).squeeze(1)  # B, H, W
        pred_patches_mask[start_idx: end_idx] = test_patch_batch_mask.detach().cpu()

    ori_h, ori_w = ori_img.shape[:2]  # (5412, 5435)
    scale_h = round(ori_h / cfg.DATA.DOWN_SAMPLE_RATE_TEST)  # 1353
    scale_w = round(ori_w / cfg.DATA.DOWN_SAMPLE_RATE_TEST)  # 1359
    # 合并成单张图片
    single_image_pred_mask_binary, single_image_pred_mask_probas = merge_patch(
                    pred_patches_mask,
                    h=scale_h, w=scale_w,
                    stride=cfg.DATA.DENSE_CROP_STRIDE,
                    merge_method=cfg.DATA.DENSE_CROP_MERGE_METHOD)
    single_image_pred_mask_binary = resize_ndarry(single_image_pred_mask_binary, ori_w, ori_h)  # H, W
    pred_masks_binary.append(single_image_pred_mask_binary[None, :, :])  # 1, H, W
    return pred_masks_binary


def inference_single_resize(dataloader, image, gt_mask, index, loss_func, model, device, cfg, record_loss):
    raise Exception("还没写")
    """
    func:
        裁剪
    return: 
        gt_masks 真实的 mask
        inference_loss 
        pred_masks_binary 预测的mask
    """
    # batch size for dense crop is 1
    # 真实 mask 整张图
    gt_masks = []
    gt_mask_ori = dataloader.dataset.get_mask(index[0])
    gt_masks.append(gt_mask_ori)
    pred_masks_binary = []
    inference_loss = []
    
    B, C, H, W = image.shape
    image = image.to(device)
    pred_mask = model(image).squeeze(1)  # B, H, W
    if record_loss:
        gt_mask = gt_mask.unsqueeze(1)  # (B H W) -> (B 1 H W)
        the_pred_mask = pred_mask.unsqueeze(1) # (B H W) -> (B 1 H W)
        loss = loss_func(gt_mask.cuda(), the_pred_mask)
        inference_loss.append(float(loss))

    ori_h, ori_w = gt_mask_ori.shape[:2]  # (5412, 5435)

    scale = min(H/ori_h, W/ori_w)
    new_h, new_w = int(ori_h*scale+0.5), int(ori_w*scale+0.5)  # 注意和 cv2 出来的是不是一样
    padding_h, padding_w = (H - new_h)//2, (W - new_w)//2

    pred_mask = pred_mask.detach().cpu().numpy()
    valid_h = slice(padding_h, padding_h+new_h)
    valid_w = slice(padding_w, padding_w+new_w)
    new_pred_mask = pred_mask[:, valid_h, valid_w]  # 去掉填充0的区域
    new_pred_mask = np.squeeze(new_pred_mask, axis=0)  # (B, H, W) -> (H, W)



    ############# 先阈值再放大
    new_pred_mask = np.where(new_pred_mask > 0.5, 1, 0).astype(np.uint8)
    new_pred_mask_large = resize_ndarry(new_pred_mask, ori_w, ori_h)  # 放大
    pred_masks_binary.append(new_pred_mask_large[None, :, :])  # 相当于加了一个维度
    #############

    ############# 先放大再阈值
    # new_pred_mask_large = resize_ndarry(new_pred_mask, ori_w, ori_h)  # 放大
    # single_image_pred_mask_binary = np.where(new_pred_mask_large > 0.5, 1, 0).astype(np.uint8)
    # pred_masks_binary.append(single_image_pred_mask_binary[None, :, :])  # 相当于加了一个维度
    #############
    return gt_masks, inference_loss, pred_masks_binary