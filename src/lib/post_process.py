import numpy as np
import cv2

def valid_patch(patch, threshold=160):
    """
    检查是不是合格的样本，不是白噪声
    在阈值以下的，全都是白噪声
    """

    h, w = patch.shape[:2]
    size = int(w/16)

    for m in range(16):
        for n in range(16):
            x, y = m*size, n*size
            mean = patch[y: y+size, x:x+size].mean()
            if mean > threshold:
                return True
    # 意思是这么多块 16*16，至少有一块要大于阈值吧
    return False

def resize_ndarry(img, width, height):
    """
    resize ndarray img to given width and height
    """
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return img

def remove_small_region(pred_mask, th=500):
    contours_mask, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    left = [cont for cont in contours_mask if cv2.contourArea(cont) > th]
    canvas = np.zeros_like(pred_mask, dtype=np.uint8)
    cv2.drawContours(canvas, left, -1, 1, -1)
    return canvas



def get_grid(img_size, crop_size, crop_stride):
    res = []
    cur = 0
    while True:
        x2 = min(cur + crop_size, img_size)
        x1 = max(0, x2 - crop_size)
        res.append([x1, x2])
        if x2 == img_size:
            break
        cur += crop_stride
    return res


def merge_patch(patches, h, w, stride=256, merge_method='or', binary_th=0.5):
    """
    merge patches split from dense crop
    :param patches: patches cropped from a single image, tensor of shape [patch_nums, h, w]
    :param h: single image's height
    :param w: width
    :param stride: crop stride
    :param merge_method: how to merge overlapping part, logical or, logical and, probability mean
    :param binary_th:
    :return: merged image, [binary mask, probability map] if using prob_mean, else return [binary, binary]
    """
    patches = np.where(patches > binary_th, 1, 0).astype(np.uint8)

    single_image_pred_mask = np.zeros((h, w), dtype=np.uint8)

    patch_h, patch_w = patches[0].shape[:2]

    h_grid = get_grid(h, patch_h, stride)
    w_grid = get_grid(w, patch_w, stride)

    ovlap_h = patch_h - stride
    ovlap_w = patch_w - stride
    idx = 0

    for y1, y2 in h_grid:
        for x1, x2 in w_grid:
            # patch in range [0, valid_w/h] is valid, otherwise is 0, which is filled for batch inference
            valid_w = x2 - x1
            valid_h = y2 - y1
            # left top
            if x1 == 0 and y1 == 0:
                new_part_x_global, new_part_y_global = slice(x1, x2), slice(y1, y2)
                new_part_x_patch, new_part_y_patch = slice(0, valid_w), slice(0, valid_h)

            # left
            elif x1 == 0 and y1 != 0:
                new_part_x_global, new_part_y_global = slice(x1, x2), slice(y1 + ovlap_h, y2)
                new_part_x_patch, new_part_y_patch = slice(0, valid_w), slice(ovlap_h, valid_h)
            # top
            elif x1 != 0 and y1 == 0:
                new_part_x_global, new_part_y_global = slice(x1 + ovlap_w, x2), slice(y1, y2)
                new_part_x_patch, new_part_y_patch = slice(ovlap_w, valid_w), slice(0, valid_h)
            # central
            else:
                new_part_x_global, new_part_y_global = slice(x1 + ovlap_w, x2), slice(y1 + ovlap_h, y2)
                new_part_x_patch, new_part_y_patch = slice(ovlap_w, valid_w), slice(ovlap_h, valid_h)
            
            np.logical_or(single_image_pred_mask[y1:y2, x1:x2],
                        patches[idx, :valid_h, :valid_w],
                        out=single_image_pred_mask[y1:y2, x1:x2])

            single_image_pred_mask[new_part_y_global, new_part_x_global] = patches[idx][new_part_y_patch, new_part_x_patch]
            idx += 1
    return single_image_pred_mask, single_image_pred_mask