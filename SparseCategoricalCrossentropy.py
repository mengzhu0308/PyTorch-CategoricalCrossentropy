#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/30 13:38
@File:          SparseCategoricalCrossentropy.py
'''

import torch
from torch import nn

class SparseCategoricalCrossentropy(nn.Module):
    def __init__(self):
        super(SparseCategoricalCrossentropy, self).__init__()

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        return -torch.sum(torch.log(y_pred[torch.arange(batch_size), y_true] + 1e-7)) / batch_size