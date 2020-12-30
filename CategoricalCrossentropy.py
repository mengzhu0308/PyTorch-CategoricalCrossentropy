#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/30 13:38
@File:          CategoricalCrossentropy.py
'''

import torch
from torch import nn

class CategoricalCrossentropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossentropy, self).__init__()

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        return -torch.sum(y_true * torch.log(y_pred + 1e-7)) / batch_size