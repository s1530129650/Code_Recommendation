#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: train.py
@time: 2019/4/25 13:47
shuffle
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
#use_gpu = True

##1. parameters setting
if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 1200
    EMBEDDING_type = 300
    HIDDEN_SIZE = 1500
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
'''
arr=np.load(r"../data/python/vocabulary.npz")
value_vocab = arr['value_vocab'].item()
type_vocab = arr['type_vocab'].item()
'''
with np.load(r"../data/python/vocabulary.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()


with open('../data/python/training.pickle', 'rb') as f:
    data = pickle.load(f)
len_data = data.length

data_loader = DataLoader(data, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)


now = time.time()
print("data loading",now-time_start)



#3.model init
model = main_model(len(value_vocab), EMBEDDING_value, len(type_vocab), EMBEDDING_type, HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW).to(device)
loss_function = nn.NLLLoss()
learning_rate = 0.01
decay = 0.6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
clip = 5
nn.utils.clip_grad_norm_(model.parameters(), clip)
losses = []

staring_training = time.time()
print("staring training ",staring_training-time_start)


## 4 training
num_epochs=5

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    total_loss = 0
    print_loss_total = 0  # Reset every print_every

    #data_loder = input_type, input_value,input_length ,target,parent

    for i, data4 in enumerate(data_loader,0):

        start = time.time()
        # step1 init
        optimizer.zero_grad()
        # step 2 prepare the data
        '''
        #if shuffle =true
        print("data4[2]",data4[2])
        print("data4[2]", data4[2].shape)
        print("data4[3]", data4[3])
        print("data4[3]", data4[3].shape)
        '''
        Z = zip(data4[2], data4[0], data4[1], data4[3],data4[4])
        Z1 = sorted(list(Z),key = lambda x:x[0],reverse=True )
        input_len, input1,input2, target, parent = zip(*Z1)
        input = [torch.cat((input1),0).view(BATCH_SIZE,-1).to(device),torch.cat((input2),0).view(BATCH_SIZE,-1).to(device)]
        input_len = torch.tensor(list(input_len))
        target = torch.tensor(list(target),device = device)
        parent = torch.tensor(list(parent))


        # step 3 get the scorece
        yt_point = model(input,input_len, model.initHidden(), parent)


        # step 4 train
        loss = loss_function(yt_point.squeeze(0), target.to(device))

        print("yt_point", yt_point.shape, yt_point)
        print("yt_point_seq", yt_point.squeeze(0).shape)
        print("target", target.shape, target)
        print(loss.item())


        loss.backward()
        optimizer.step()

        # loss
        total_loss += loss.item()
        #topv, topi = yt_point.data.topk(1)
        #eval_index = topi.squeeze().detach()
        # print(eval_index)
        end = time.time()
        #print(i, "batch time spend", end - start)

    now = time.time()

    print('epoch = %d  time spend:%s  loss%.8f' % (
    epoch + 1, (now - time_start)/60, total_loss ))
    losses.append(total_loss )

print(losses)
torch.save(model.state_dict(), 'params_lstm_attn.pkl')
model.load_state_dict(torch.load('params_lstm_attn.pkl'))






