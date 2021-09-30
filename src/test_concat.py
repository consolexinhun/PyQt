import argparse
import logging
import os

from torch.utils.data import DataLoader

from config import cfg
from solver import Solver
from transform.build import concat_build_transforms
from model.build import concat_build_model
from loss.build import build_loss
from optimizer.build import build_optimizer
from dataset import Colonoscopy, collate_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="配置文件")

    # 加载参数
    args = parser.parse_args()
    if args.config is not None:
        # import ipdb; ipdb.set_trace()
        cfg.merge_from_file(args.config)
    cfg.freeze()

    # 日志文件
    logging.basicConfig(filename=os.path.join(cfg.OUTPUT_DIR, "test.log"), 
                        filemode="w", level=logging.INFO)
    logger = logging.getLogger("test log")
    logger.addHandler(logging.StreamHandler())
    logger.info("配置文件如下：")
    logger.info(cfg)


    transform_test = concat_build_transforms(cfg, is_train=False)
    dataset_test = Colonoscopy(split_file=cfg.DATA.DATASET_TEST,
                                img_root=cfg.DATA.DATA_ROOT,
                                transforms=transform_test)
                                
    dataloader_test = DataLoader(dataset_test,
                                batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                num_workers=0,
                                collate_fn=collate_fn)

    model = concat_build_model(cfg)
    loss = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    solver = Solver(cfg, loss, model, optimizer, logger, cfg.OUTPUT_DIR, is_train=False)
    solver.test(dataloader_test=dataloader_test,  tta=False)



    

