from functools import partial

import torch
import torch.nn as nn
from torchmetrics.functional.image import spatial_correlation_coefficient as scc

# TODO
# scc_mask = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device="cuda")
# SCC = partial(scc, hp_filter=scc_mask, window_size=8)


def oneminusx(x):
    return 1 - x


class DoubleLoss(nn.Module):
    """fmerge(f1(loss1(pred,target)), f2(loss2(pred,target)))"""

    def __init__(self, loss1, loss2, f1=None, f2=None, merge_function=torch.sum):
        super().__init__()
        if isinstance(loss2, str):
            if loss2 == "SCC":
                loss2 = scc
        if isinstance(loss1, str):
            if loss1 == "SCC":
                loss1 = scc
        self.loss1 = loss1
        self.loss2 = loss2
        self.f1 = f1
        self.f2 = f2
        self.merge_function = merge_function

    def forward(self, inputs, targets):

        l1 = self.loss1(inputs, targets)
        l2 = self.loss2(inputs, targets)
        if self.f1:
            l1 = self.f1(l1)
        if self.f2:
            l2 = self.f2(l2)

        return self.merge_function(l1, l2)
