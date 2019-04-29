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
import numpy as np

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


with open('../data/python/training1_50k.pickle', 'rb') as f:
    data_train = pickle.load(f)
len_train = data_train.length
print(len_train)

with open('../data/python/eval1_50k.pickle', 'rb') as f:
    data_eval = pickle.load(f)
len_eval = data_eval.length
print(len_eval)

#data_loader_train = DataLoader(data_train, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
#data_loader = rnn_utils.pad_sequence(data_loader_train, batch_first=True, padding_value=0)
#data_loder = 0 input_type, 1 input_value,2 input_length ,3 target,4  parent

countList = np.array([])
for i, data4 in enumerate(data_train ,0):
    print('iter {}'.format(i))

    #countList = np.append(countList, data4[2].numpy())
    print('-' * 10)
    print(data4[3])
    print(data4[2])
    print('-' * 10)
    if i == 20: break
'''
max_value = np.max(countList)
min_value = np.min(countList)
avg_value = np.mean(countList)
var_value =  np.var(countList)
std_value = np.std(countList,ddof=1)
MT2000 = np.sum(countList > 2000)
print("taining set")
print("max:",max_value)
print("min:",min_value)
print("avg:",avg_value)
print("var:",var_value)
print("std:",std_value)
print("the len of sequence more than 2000:",MT2000 ,MT2000 / countList.size)
'''