import torch
import torch.nn as nn


class DoubleLoss(nn.Module):
    """fmerge(f1(loss1(pred,target)), f2(loss2(pred,target)))"""

    def __init__(self, loss1, loss2, f1=None, f2=None, merge_function=torch.sum):
        super().__init__()
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
