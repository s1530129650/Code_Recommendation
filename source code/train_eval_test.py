#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: train_eval_test.py
@time: 2019/4/28 14:45
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  pickle
import random
import numpy as np
import time
torch.manual_seed(1)
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data

from model import main_model

time_start = time.time()

use_gpu = False
use_gpu = True

##1. parameters setting
if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 50000
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 512
    EMBEDDING_type = 256
    HIDDEN_SIZE = 512
    BATCH_SIZE = 10

else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 2
    EMBEDDING_type = 3
    HIDDEN_SIZE = 5
    BATCH_SIZE = 2

# 2.data loading
class MyData(data.Dataset):
    def __init__(self,data_seq, input_value, input_type, target, parent):
        self.input_value = input_value
        self.input_type = input_type
        self.target = target
        self.parent = parent
        self.length = len(self.target)
        self.data_length = [len(sq) for sq in data_seq]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.input_type[idx],self.input_value[idx], self.data_length[idx], self.target[idx], self.parent[idx]
#vocabulary
with np.load(r"../data/python/vocabulary_trainAndeval_50k.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()

#train

with np.load(r"../data/python/train.npz", allow_pickle=True) as arr:
    input_value = arr['input_value']
    input_type = arr['input_type']
    parent = arr['parent']
    target = arr['target']

'''
x_train = rnn_utils.pad_sequence(input_value_train, batch_first=True)
y_train = rnn_utils.pad_sequence(input_type_train, batch_first=True)
dataAll_train = MyData(input_value_train, x_train, y_train, target_train, parent_train)

'''

print(  input_value)
