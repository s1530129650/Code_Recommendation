#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: chech_input.py
@time: 2019/4/26 15:16
"""
import pickle
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data

use_gpu = False
use_gpu = True

##1. parameters setting
if use_gpu:
    device = torch.device("cuda")
    BATCH_SIZE = 10

else:
    device = torch.device("cpu")
    BATCH_SIZE = 1


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


with open('../data/python/training.pickle', 'rb') as f:
    data_train = pickle.load(f)
len_train = data_train.length
data_loader_train = DataLoader(data_train, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
#data_loader = rnn_utils.pad_sequence(data_loader_train, batch_first=True, padding_value=0)
#data_loder = 0 input_type, 1 input_value,2 input_length ,3 target,4  parent
for i, data4 in enumerate(data_loader_train ,0):
    print('iter {}'.format(i))

    print(data4[3])
    print('-' * 10)
