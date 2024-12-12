"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR, PolynomialLRDecay


def build_optimizer(optim, model):
    name = optim.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=optim.BASE_LR,
            weight_decay=optim.WEIGHT_DECAY,
            **optim.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_optimizer_dis(cfg, model):
    name = cfg.OPTIMIZER_DIS.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER_DIS.BASE_LR,
            weight_decay=cfg.OPTIMIZER_DIS.WEIGHT_DECAY,
            **cfg.OPTIMIZER_DIS.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = ClipLR(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)

    return scheduler

def build_scheduler_dis(cfg, optimizer):
    name = cfg.SCHEDULER_DIS.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif name == 'PolyLR':
        scheduler = PolynomialLRDecay(optimizer, cfg.SCHEDULER.MAX_ITERATION, **cfg.SCHEDULER_DIS.get(name, dict()))
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER_DIS.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    return scheduler
