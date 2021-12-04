import wandb
from torch import nn
from torch.nn import functional as F

import pandas as pd
import torch
import numpy as np
import os
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_weighted_ce(distribution):
    class WeightedCE(nn.CrossEntropyLoss):

        def __init__(self) -> None:
            # Default: weight=None, size_average=None, ignore_index=-100,
            # reduce=None, reduction'mean'
            paras = {'weight': None,
                     'size_average': None,
                     'ignore_index': -100,
                     'reduce': None,
                     'reduction': 'none'}
            super(WeightedCE, self).__init__(**paras)
            self.paras = paras
            self.distribution = (distribution / distribution.sum()).to('cuda')

        def forward(self, input: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
            loss = F.nll_loss(F.log_softmax(input, 1), target, **self.paras)
            return torch.dot(loss, self.distribution)

    return WeightedCE


def init_logger(config: dict):
    wandb.init(project="AdaDL", entity="lester")
    wandb.config = config
    wandb.run.name = '_'.join([f'{k}{v}' for k, v in config.items()])


def log(logs, step=1):
    wandb.log(logs, step=step)
