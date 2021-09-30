from torch.optim import SGD, Adam

def build_optimizer(model, cfg):
    name = cfg.MODEL.OPTIMIZER
    lr = cfg.SOLVER.LR
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    if name == "Adam":
        return Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else :
        raise Exception("没有这个优化器")

