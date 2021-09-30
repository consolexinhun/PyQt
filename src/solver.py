import os
import datetime
import time
from collections import defaultdict

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import imageio

from lib.checkpoint import CheckPointer
from lib.evaluation import do_evaluation, do_test_evaluation
from lib.inference import inference_dense_crop, inference_resize
from lib.inference_single import inference_single_dense_crop, inference_single_resize, inference_single_dense_crop_progress


class Solver:
    def __init__(self, cfg, loss=None, model=None, optimizer=None, logger=None, output_dir=None, is_train=True):
        self.cfg = cfg
        self.loss = loss
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.is_train = is_train
        self.logger = logger

        self._initialize_model_device()

    def _initialize_model_device(self):
        """
        初始化设备
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

            if torch.cuda.device_count() > 1:
                self.logger.info(f'Using{torch.cuda.device_count()}GPUs')
                self.model = nn.DataParallel(self.model)
            else:
                self.logger.info('Using Single GPU')
        else:
            self.device = torch.device('cpu')
            self.logger.info('cuda is not available, using cpu')

        self.model.to(self.device)

    def get_tensorboard_dir(self):
        """
        获取 tensorboard 日志文件的 目录！！，如果有就恢复
        """
        tb_log_dir = os.path.join(self.output_dir, datetime.datetime.now().strftime('%b%d_%H'))
        if os.path.exists(tb_log_dir):
            self.logger.info(f'Resume from exiting tensorboard log dir: {tb_log_dir}')
        else:
            self.logger.info(f'Creating new tensorboard log dir: {tb_log_dir}')
        return tb_log_dir
    
    def train(self, dataloader_train, dataloader_valid=None, dataloader_test=None):
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, verbose=True, patience=10)

        # 断点加载
        checkpointer = CheckPointer(self.model, self.logger, self.optimizer, scheduler, self.output_dir)
        extra_checkpoint_data = checkpointer.load(self.cfg.MODEL.RESUME, need_resume=False)

        # 训练轮数
        max_epochs = self.cfg.SOLVER.EPOCHS
        start_epoch = extra_checkpoint_data.get("epoch", 0)

        # tensorboard 日志
        tb_log_dir = self.get_tensorboard_dir()
        # tb_writer = SummaryWriter(log_dir=tb_log_dir, purge_step=start_epoch)

        save_arguments = {}  # 暂时只保存了 epoch

        best_valid_dice =  0
        best_valid_auc = 0


        for epoch in range(start_epoch, max_epochs):
            return_dict = self.train_one_epoch(epoch, dataloader_train, self.optimizer)
            ## train_losses, eval_gt_mask, eval_dt_mask
            
            loss_one_epoch = np.mean(return_dict['train_losses'])
            scheduler.step(loss_one_epoch)

            save_arguments['epoch'] = epoch + 1

            current_lr = self.optimizer.param_groups[0]['lr']

            now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.logger.info(f'{now_time}, Epoch: {epoch:<3d} loss: {loss_one_epoch:.4f} lr: {current_lr:.6f}')

            # 曲线画到 tensorboard 中
            # tb_writer.add_scalar('loss/train', loss_one_epoch, global_step=epoch)
            # tb_writer.add_scalar('lr', current_lr, global_step=epoch)

            # 训练集的评估
            with torch.no_grad():
                train_metrics = do_evaluation(return_dict['eval_gt_mask'], return_dict['eval_dt_mask'])
                # for key, value in train_metrics.items():
                    # tb_writer.add_scalar(f'{key}/train', value, global_step=epoch)

                
            if epoch % self.cfg.SOLVER.SAVE_INTERVAL_EPOCH == 0:
                if dataloader_valid is not None:
                    gt_masks, pred_masks, valid_loss = self.inference(dataloader_valid,
                                                                        epoch, post_process=False,
                                                                        record_loss=True)
                    # 注意，这里关了后处理

                    # 验证集上的指标，注意是整张图完整的情况
                    valid_metrics = do_evaluation(gt_masks, pred_masks)
                    # for key, value in valid_metrics.items():
                    #     tb_writer.add_scalar(f'metrics_whole_{key}/valid', value, global_step=epoch)
                    # tb_writer.add_scalar('loss/valid', valid_loss, global_step=epoch)


                    self.logger.info(f'Valid result: {valid_metrics}' )
                    self.logger.info(f'Valid loss: {valid_loss}' )

                    # 2. save best valid dice point 保存最好的模型
                    valid_dice_1 = valid_metrics['dice']
                    if valid_dice_1 > best_valid_dice:
                        best_valid_dice = valid_dice_1
                        self.logger.info('Saving best valid dice point')
                        checkpointer.save('best_valid_dice', **save_arguments)
                    valid_auc = valid_metrics['auc']
                    if valid_auc > best_valid_auc:
                        best_valid_auc = valid_auc
                        self.logger.info('Saving best valid auc point')
                        checkpointer.save('best_valid_auc', **save_arguments)

                    del gt_masks, pred_masks, valid_loss

                    # 4. save model
                checkpointer.save("model_epoch_{:03d}".format(epoch), **save_arguments)

    
    def train_one_epoch(self, epoch, dataloader_train, optimizer):
        """
        return:
            dict(
                "eval_gt_mask": (B, 1, H, W)
                "eval_dt_mask": (B, 1, H, W)
                "train_losses": []
            )
        """
        self.model.train()
        train_losses = []
        return_dict = {}

        for i, (image, mask, _) in enumerate(tqdm(dataloader_train, desc='Training epoch {:4d}'.format(epoch), ncols=0)):
            mask = mask.unsqueeze(1)  # B, H, W -> (B, 1, H, W)

            image = image.to(self.device)
            mask = mask.to(self.device)
            logits = self.model(image)   # logits: tensor(B, 1, H, W)


            loss = self.loss(mask, logits) 
            train_losses.append(float(loss))

            # back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 只需要记录最后一次的结果就行了
            if i == len(dataloader_train)-1:
                with torch.no_grad():
                    self.model.eval()
                    dt_masks = self.model(image)  # (B, 1, H, W)

                    return_dict['eval_gt_mask'] = [np.array(x.cpu()) for x in mask]
                    return_dict['eval_dt_mask'] = [np.array(x.detach().cpu()) for x in dt_masks]
                    self.model.train()

        return_dict['train_losses'] = train_losses

        return return_dict

    def inference(self, dataloader, epoch=None, post_process=False, record_loss=False, is_tta=False):
        """
        inference on valid or test dataset
        use sliding window if cfg.DATA.CROP_METHOD_TEST is set to 'dense'
        :param dataloader:
        :param epoch:
        :param post_process: whether do post process, eg. fill hole
        :param record_loss: whether to calculate loss
        :param is_tta: whether use test time augmentation
        :return: gt_masks, pred_masks, vis_images, list of ndarray
        """
        self.model.eval()

        pred_masks_binary = []  # 预测的二值mask 整张图
        gt_masks = []           # 真实的二值mask 整张图
        inference_loss = []     # 预测的loss

        epoch = 'test' if epoch is None else epoch  # 验证是 epoch，测试是 test

        with torch.no_grad():
            for image, gt_mask, index in tqdm(dataloader, desc='Epoch {}, inference....'.format(epoch)):
                # image: B, 3, H, W
                # mask: B, H, W
                # index: [] 只有1个数，对应其索引
                if self.cfg.DATA.CROP_METHOD_TEST == 'DenseCrop':
                    gt_mask, loss, pred_mask = inference_dense_crop(dataloader, image, gt_mask, index, self.loss, self.model, self.device, self.cfg, record_loss)
                    gt_masks.extend(gt_mask)
                    inference_loss.extend(loss)
                    pred_masks_binary.extend(pred_mask)
                elif self.cfg.DATA.CROP_METHOD_TEST == "Resize":
                    gt_mask, loss, pred_mask = inference_resize(dataloader, image, gt_mask, index, self.loss, self.model, self.device, self.cfg, record_loss)
                    gt_masks.extend(gt_mask)
                    inference_loss.extend(loss)
                    pred_masks_binary.extend(pred_mask)
                else:
                    print('use dense crop or resize in test and validation')
                    raise ValueError

        if record_loss:
            return gt_masks, pred_masks_binary, np.mean(inference_loss)
        else:
            return gt_masks, pred_masks_binary


    def vis_img(self, img, gt_mask, pred_mask):
        """
        可视化 gt pred 的 mask 的重叠区域
        """

        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask.squeeze(0)
        
        # img = img.transpose(1, 2, 0).astype(np.float32)  # C, H, W -> H, W, C
        pred_mask[pred_mask <= 0.5] = 0
        pred_mask[pred_mask > 0.5] = 1
        pred_mask.astype(np.uint8)

        gt_mask = gt_mask.astype(np.uint8)

        img[..., 0] = np.where(pred_mask == 1, 255, img[..., 0])  # R 通道表示预测的
        img[..., 1] = np.where(gt_mask == 1, 255, img[..., 1])  # G 通道表示绿色的
        # 如果两个有重叠，红 + 绿 = 黄

        img = img[..., (2, 1, 0)]  # RGB -> BGR 为了保存
        return img


    def test(self, dataloader_test, tta):
        """
        test on test data set
        :param dataloader_test:
        :param tta:
        :return:
        """
        checkpointer = CheckPointer(self.model, self.logger, self.optimizer, save_dir=self.output_dir)
        checkpointer.load(resume_iter=self.cfg.MODEL.RESUME, need_resume=True, best_valid=True)

        self.model.eval()

        
        vis_root = os.path.join(self.output_dir, 'vis')
        if not os.path.exists(vis_root):
            os.makedirs(vis_root)
        self.logger.info(f'Saving visualization image...to{vis_root}')

        with torch.no_grad(), open(os.path.join(vis_root, 'result.csv'), 'w') as f:
            result_list = defaultdict(list)  # 创建一个字典，字典的值默认是 空列表[]

            f.write('name, dice, gt, pred\n')
            
            for image, gt_mask, index in tqdm(dataloader_test, desc='test, inference....'):
                # image: B, 3, H, W
                # mask: B, H, W
                # index: [] 只有1个数，对应其索引
                # start = time.time()
                if self.cfg.DATA.CROP_METHOD_TEST == 'DenseCrop':
                    gt_mask, loss, pred_mask = inference_dense_crop(dataloader_test, image, gt_mask, index, self.loss, self.model, self.device, self.cfg, False)
                elif self.cfg.DATA.CROP_METHOD_TEST == "Resize":
                    gt_mask, loss, pred_mask = inference_resize(dataloader_test, image, gt_mask, index, self.loss, self.model, self.device, self.cfg, False)
                
                # import ipdb; ipdb.set_trace()
                
                gt_mask = gt_mask[0]
                pred_mask = pred_mask[0]

                gt_label, pred_label, dice_1 = do_test_evaluation(gt_mask, pred_mask)
                result_list['gt_label'].append(gt_label)
                result_list['pred_label'].append(pred_label)
                result_list['dice_1'].append(dice_1)

                

                if self.cfg.SOLVER.DRAW:
                    vis_image, image_name = dataloader_test.dataset.get_img(index[0]), dataloader_test.dataset.get_name(index[0])
                    # 可视化结果
                    # import ipdb; ipdb.set_trace()
                    

                    # 保存原图 叠加 gt_mask 和 pd_mask 的文件
                    result_img = self.vis_img(vis_image, gt_mask, pred_mask)
                    save_path_img = os.path.join(vis_root, f'dice_{dice_1:.3f}_gt{gt_label}_pred{pred_label}_{image_name}.jpg')
                    cv2.imwrite(save_path_img, result_img)

                    # 保存 pd_mask 文件
                    save_path_mask = os.path.join(vis_root, f'dice_{dice_1:.3f}_gt{gt_label}_pred{pred_label}_{image_name}_mask.jpg')
                    pred_mask = pred_mask.squeeze(0)
                    pred_mask[pred_mask > 0.5] = 1
                    pred_mask[pred_mask <= 0.5] = 0
                    mask_to_save = (pred_mask * 255).astype(np.uint8)
                    cv2.imwrite(save_path_mask, mask_to_save)

                    f.write('{}, {}, {}, {}\n'.format(image_name, dice_1, gt_label, pred_label))


            dice_1 = [x for x in result_list['dice_1'] if x >= 0]  # 只计算阳性类别的 Dice
            dice = np.mean(dice_1) if len(dice_1) > 0 else 0  # 所有样本 的 Dice 的平均
            try:
                auc = roc_auc_score(result_list['gt_label'], result_list['pred_label'])
            except ValueError:
                auc = -1
            acc = accuracy_score(result_list['gt_label'], result_list['pred_label'])
            precision = precision_score(result_list['gt_label'], result_list['pred_label'], zero_division=0)
            recall = recall_score(result_list['gt_label'], result_list['pred_label'], zero_division=0)

            result = {
                'dice': dice,
                'auc': auc,
                'acc': acc,
                'precision': precision,
                'recall': recall
            }
            self.logger.info(f'Test result:{result}')


    def test_single(self, dataloader_test):
        """
        test on test data set
        :param dataloader_test:
        :param tta:
        :return:
        """
        checkpointer = CheckPointer(self.model, self.logger, self.optimizer, save_dir=self.output_dir)
        checkpointer.load(resume_iter=self.cfg.MODEL.RESUME, need_resume=True, best_valid=True)
        
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.model.eval()
        for image, index in tqdm(dataloader_test, desc='test, inference....'):
            # image: B, 3, H, W
            # mask: B, H, W
            # index: [] 只有1个数，对应其索引
            if self.cfg.DATA.CROP_METHOD_TEST == 'DenseCrop':
                pred_mask = inference_single_dense_crop(dataloader_test, image, index, self.model, self.device, self.cfg)
            elif self.cfg.DATA.CROP_METHOD_TEST == "Resize":
                pred_mask = inference_single_resize(dataloader_test, image, index, self.model, self.device, self.cfg, False)
            
            pred_mask = pred_mask[0]

            pred_mask = pred_mask.squeeze(0)
            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0
            mask_to_save = (pred_mask * 255).astype(np.uint8)

            # cv2.imwrite("tmp.jpg", mask_to_save)
            return mask_to_save

    def test_single_progress(self, dataloader_test, progress):
        """
        test on test data set
        :param dataloader_test:
        :param tta:
        :return:
        """
        checkpointer = CheckPointer(self.model, self.logger, self.optimizer, save_dir=self.output_dir)
        checkpointer.load(resume_iter=self.cfg.MODEL.RESUME, need_resume=True, best_valid=True)

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.model.eval()
        for image, index in tqdm(dataloader_test, desc='test, inference....'):
            # image: B, 3, H, W
            # mask: B, H, W
            # index: [] 只有1个数，对应其索引
            if self.cfg.DATA.CROP_METHOD_TEST == 'DenseCrop':
                # pred_mask = inference_single_dense_crop(dataloader_test, image, index, self.model, self.device,
                #                                         self.cfg)
                pred_mask = inference_single_dense_crop_progress(dataloader_test, image, index, self.model, self.device,
                                                        self.cfg, progress)

            elif self.cfg.DATA.CROP_METHOD_TEST == "Resize":
                pred_mask = inference_single_resize(dataloader_test, image, index, self.model, self.device, self.cfg,
                                                    False)

            pred_mask = pred_mask[0]

            pred_mask = pred_mask.squeeze(0)
            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0
            mask_to_save = (pred_mask * 255).astype(np.uint8)

            return mask_to_save
