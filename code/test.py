#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence
@contact: 123@qq.com
@site:
@software: PyCharm
@file: main3.py
@time: 2019/3/25 21:09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dfs import dfs_AST
import json
import random

import time
time_start=time.time()

str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"


torch.manual_seed(1)

EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5
CONTEXT_WINDOW = 3
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 3
max_vocab_size = 1000
'''
data = [[ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num","value":"1"} ]]
'''
data = []
with open(str+"\data\python\python100k_train.json",'r') as load_f:
    data1 = load_f.readlines()
for i in range(1):
    content = json.loads(data1[i])
    data.append(content)
print("data:",data)

data_flatten = []
for i  in range(len(data)):
    data_flatten .append(dfs_AST(data[i], 0))

print("data_flatten:",data_flatten)

print("data:",data)



