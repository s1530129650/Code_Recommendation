#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: model_batch.py
@time: 2019/4/24 22:15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import time
import argparse

'''
def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=10,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=128,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.01,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='in case of gradient explosion')
    return p.parse_args()
'''

time_start = time.time()
torch.manual_seed(1)

use_gpu = False
#use_gpu = True
if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
    BATCH_SIZE = 128
else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100
    BATCH_SIZE = 2

##1. data loading
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


# read
with open('../data/python/training.pickle', 'rb') as f:
    data = pickle.load(f)

data_loader = DataLoader(data, batch_size= BATCH_SIZE, shuffle=True)
'''
batch_x = iter(data_loader).next()
print(batch_x[0])
'''

for i,data2 in enumerate(data_loader,0):
    print(i)
    print(data2)








