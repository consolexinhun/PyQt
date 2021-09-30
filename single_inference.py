"""
单张图片的推断，来给客户端用的
"""
import os, sys, logging
sys.path.append("src")

from torch.utils.data import DataLoader

from config import cfg
from solver import Solver
from transform.build import build_transforms
from model.build import build_model
from dataset import ColonoscopySingle, collate_fn_single


cfg.merge_from_file("config/psp/pspnet3.yaml")
cfg.freeze()

transform_test = build_transforms(cfg, is_train=False)

model = build_model(cfg)

logging.basicConfig(filename="single inference.log", 
                    filemode="w", level=logging.ERROR)
logger = logging.getLogger("single inference log")
logger.addHandler(logging.StreamHandler())
logger.info("配置文件如下：")
logger.info(cfg)

solver = Solver(cfg, None, model, None, logger, cfg.OUTPUT_DIR, is_train=False)


def do_inference(img_path):
    dataset_test = ColonoscopySingle(
                                img_path=img_path,
                                transforms=transform_test)
    dataloader_test = DataLoader(dataset_test,
                                batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                num_workers=0,
                                collate_fn=collate_fn_single)
    res = solver.test_single(dataloader_test=dataloader_test)
    return res


def do_inference_progress(img_path, progress):
    dataset_test = ColonoscopySingle(
                                img_path=img_path,
                                transforms=transform_test)
    dataloader_test = DataLoader(dataset_test,
                                batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                num_workers=0,
                                collate_fn=collate_fn_single)
    res = solver.test_single_progress(dataloader_test=dataloader_test, progress=progress)
    return res
# 主要解决不同步显示的问题


if __name__ == "__main__":
    do_inference("D:/D201709874_2019-05-21 12_42_42-lv1-31315-17194-2766-2641.jpg")

