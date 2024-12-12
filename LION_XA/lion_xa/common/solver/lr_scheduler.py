from __future__ import division
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

class ClipLR(object):
    """Clip the learning rate of a given scheduler.
    Same interfaces of _LRScheduler should be implemented.

    Args:
        scheduler (_LRScheduler): an instance of _LRScheduler.
        min_lr (float): minimum learning rate.

    """

    def __init__(self, scheduler, min_lr=1e-5):
        assert isinstance(scheduler, _LRScheduler)
        self.scheduler = scheduler
        self.min_lr = min_lr

    def get_lr(self):
        return [max(self.min_lr, lr) for lr in self.scheduler.get_lr()]

    def __getattr__(self, item):
        if hasattr(self.scheduler, item):
            return getattr(self.scheduler, item)
        else:
            return getattr(self, item)

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0, power=0.9):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr