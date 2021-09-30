import os

from yacs.config import CfgNode as CN


cfg = CN()

cfg.PRO_ROOT = os.path.abspath(os.path.join(os.getcwd()))  # 项目根目录
cfg.OUTPUT_DIR = os.path.join(cfg.PRO_ROOT, 'output')  # 输出目录，包括可视化结果，权重文件，指标曲线

############################################################## 数据

cfg.DATA = CN()
# 训练 验证 测试
cfg.DATA.DATASET_TRAIN = './data/data_split/0_train.txt' # os.path.join(cfg.PRO_ROOT, "data/data_split/0_train.txt")  # './data/data_split/0_train.txt'
cfg.DATA.DATASET_VALID = './data/data_split/0_valid.txt' # os.path.join(cfg.PRO_ROOT, "data/data_split/0_valid.txt")  # './data/data_split/0_valid.txt'
cfg.DATA.DATASET_TEST = './data/data_split/0_valid.txt' # os.path.join(cfg.PRO_ROOT, "data/data_split/0_valid.txt")  # './data/data_split/0_valid.txt'

cfg.DATA.DATA_ROOT = "./data/raw_data"  # 原始数据


cfg.DATA.NUM_CLS = 2 # 输出的类别

## 数据处理

# 输入图片大小
cfg.DATA.SIZE_TRAIN = 512
cfg.DATA.SIZE_TEST = 512

# 裁剪方式
# 训练裁剪
cfg.DATA.CROP_METHOD_TRAIN = "RandomSampleCrop"  # RandomSampleCrop or Resize
# 测试裁剪
#########################  DenseCrop
cfg.DATA.CROP_METHOD_TEST = "DenseCrop"  # random, grid, DenseCrop or resize
cfg.DATA.DENSE_CROP_STRIDE = 256  # 裁剪步长
cfg.DATA.DENSE_CROP_MERGE_METHOD = "or"  # "or" or "and"，合并重叠区域的方式
########################

cfg.DATA.RANDOM_ROTATE = True  # 随机旋转 


# 下采样率
cfg.DATA.DOWN_SAMPLE_RATE = 4.0  # 训练的
cfg.DATA.DOWN_SAMPLE_RATE_TEST = 4.0  # 验证的

# 正负样本的均值和方差
cfg.DATA.MEAN = (200.88868021, 173.87053633, 205.59562856)
cfg.DATA.STD = (54.10893796, 76.54162457, 44.94489184)

################################################################# 训练过程

cfg.SOLVER = CN()
# 对于random, grid crop, 每个 batch 补丁总数是 batch size * patch num per img 
# 对于 resize，每个 batch 总补丁数是 batch size
# 对于 dense crop，batch size 必须是 1，补丁数随着图像大小变化

# 每个 batch 图片数量
cfg.SOLVER.BATCH_SIZE_TRAIN = 2
cfg.SOLVER.BATCH_SIZE_TEST = 1
# 每张图补丁数
cfg.SOLVER.PATCH_NUM_PER_IMG_TRAIN = 2
# cfg.SOLVER.PATCH_NUM_PER_IMG_TEST = 2

cfg.SOLVER.BATCH_SIZE_PER_IMG_TEST = 8  # 对于滑动窗口，测试过程中一个 batch 有多少个 patch

cfg.SOLVER.SAVE_INTERVAL_EPOCH = 10  # 每隔多少个 epoch 保存一次
cfg.SOLVER.EPOCHS = 100  # 训练几轮

# cfg.SOLVER.STEPS = (40, 70)  # 多步长学习率
cfg.SOLVER.LR = 0.001  # 学习率
cfg.SOLVER.WEIGHT_DECAY = 0.0001  # weight decay
cfg.SOLVER.MOMENTUM = 0.9  # 动量系数


cfg.SOLVER.DRAW = True  # 测试阶段可视化

################################################################# 模型

cfg.MODEL = CN()

# 下面这几个只针对 UNet 家族的模型
################################### UNet
cfg.MODEL.MODEL = "UNet"
cfg.MODEL.DEEP_SUP = ''
cfg.MODEL.DEEP_SUP_SE = False
cfg.MODEL.FUSION_DS = ''
cfg.MODEL.CLS_BRANCH = False
###################################

cfg.MODEL.LOSS = "DiceLoss"  # 在 lib/loss/factory 中
cfg.MODEL.OPTIMIZER = "SGD"  # 在 lib/optimizer/factory 中

cfg.MODEL.RESUME = ""  # 训练或测试阶段，090 表示恢复第 90 个 epoch 的权重
cfg.MODEL.LOSS_WEIGHT = (1.0, 1.0)  # 前景和背景的权重
# cfg.MODEL.BCE_WEIGHT = 1.0  # BCEWithLogDiceLoss
# cfg.MODEL.LOSS_FOCAL_GAMMA = 2.0  # focal loss 的 \gamma




