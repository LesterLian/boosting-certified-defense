import torch
from torch import nn
from torch.nn import functional as F


def get_weighted_ce(distribution):
    class WeightedCE(nn.CrossEntropyLoss):

        def __init__(self) -> None:
            # weight=None, size_average=None, ignore_index=-100, reduce=None, reduction'mean'
            paras = (None, None, -100, None, 'none')
            super(WeightedCE, self).__init__(*paras)
            self.paras = paras
            self.distribution = distribution.to('cuda')

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.dot(F.nll_loss(F.log_softmax(input, 1), target, *self.paras), self.distribution)

    return WeightedCE
