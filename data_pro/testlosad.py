#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: testlosad.py
@time: 2019/4/19 16:41
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  pickle
import json
import random
import numpy as np
import time
torch.manual_seed(1)
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data

BATCH_SIZE = 2

class MyData(data.Dataset):
    def __init__(self, input_value, input_type, target, parent):
        self.input_value = input_value
        self.input_type = input_type
        self.target = target
        self.parent = parent

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.input_value[idx], self.input_type[idx], self.target[idx], self.parent[idx]


# 读取
with open('../data/python/training.pickle', 'rb') as f:
    data = pickle.load(f)

data_loader = DataLoader(data, batch_size= BATCH_SIZE, shuffle=True)
batch_x = iter(data_loader).next()
print(batch_x[0])
'''
for i,data2 in enumerate(data_loader,0):
    print(i)
    print(data2)
'''



