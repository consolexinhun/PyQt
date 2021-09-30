import os
import argparse
import logging

from torch.utils.data import DataLoader

# 自定义的类
from config import cfg
from lib.seed import seed_torch
from dataset import Colonoscopy, collate_fn
from solver import Solver
from transform.build import global_build_transforms
from loss.build import build_loss
from model.build import build_model
from optimizer.build import build_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="配置文件")

    # 加载参数
    args = parser.parse_args()
    if args.config is not None:
        cfg.merge_from_file(args.config)
    cfg.freeze()

    # 创建输出文件
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # 日志文件
    logging.basicConfig(filename=os.path.join(cfg.OUTPUT_DIR, "train.log"), 
                        filemode="w", level=logging.INFO)
    logger = logging.getLogger("train log")
    logger.addHandler(logging.StreamHandler())
    logger.info("配置文件如下：")
    logger.info(cfg)

    # 设置随机种子
    seed_torch(0)

    # 数据处理

    # 训练集
    transforms_train = global_build_transforms(cfg, is_train=True)
    dataset_train = Colonoscopy(split_file=cfg.DATA.DATASET_TRAIN,
                                img_root=cfg.DATA.DATA_ROOT,
                                transforms=transforms_train)
    dataloader_train = DataLoader(dataset_train, 
                                batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                num_workers=0,
                                shuffle=True,
                                collate_fn=collate_fn)
    # 验证集
    transforms_valid = global_build_transforms(cfg, is_train=False)
    dataset_valid = Colonoscopy(split_file=cfg.DATA.DATASET_VALID,
                                img_root=cfg.DATA.DATA_ROOT,
                                transforms=transforms_valid,
                                )
    dataloader_valid = DataLoader(dataset_valid, 
                                batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                num_workers=0, 
                                collate_fn=collate_fn)

    # 损失函数，模型和优化器
    loss = build_loss(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    solver = Solver(cfg, loss, model, optimizer, logger, cfg.OUTPUT_DIR)
    solver.train(dataloader_train, dataloader_valid)
