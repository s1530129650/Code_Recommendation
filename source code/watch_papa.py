#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: watch_papa.py
@time: 2019/4/26 14:51
"""
import torch
from model import main_model
import numpy as np




use_gpu = False
use_gpu = True

##1. parameters setting
if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 300
    EMBEDDING_type = 200
    HIDDEN_SIZE = 500
    BATCH_SIZE = 100

else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 2
    EMBEDDING_type = 3
    HIDDEN_SIZE = 5
    BATCH_SIZE = 2

with np.load(r"../data/python/vocabulary_50k.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()

#model = main_model(len(value_vocab), EMBEDDING_value, len(type_vocab), EMBEDDING_type, HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW).to(device)
#model.load_state_dict(torch.load('params_lstm_attn.pkl'))

print(len(type_vocab ),len(value_vocab))
'''
for param in model.parameters():
    print(len(param))
    print(param)
    print("%%%%%%%%%%%%")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
'''