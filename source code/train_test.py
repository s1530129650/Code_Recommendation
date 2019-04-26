#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: train_test.py
@time: 2019/4/26 11:07
"""
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
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 700
    EMBEDDING_type = 300
    HIDDEN_SIZE = 1000
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

with np.load(r"../data/python/vocabulary.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()


with open('../data/python/training.pickle', 'rb') as f:
    data_train = pickle.load(f)
len_train = data_train.length
data_loader_train = DataLoader(data_train, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

with open('../data/python/eval.pickle', 'rb') as f:
    data_eval = pickle.load(f)
len_eval = data_eval.length
data_loader_eval = DataLoader(data_eval, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

now = time.time()
print("data loading",now-time_start)


##3 train
def train(model,optimizer,data_loader,loss_function):
    total_loss = 0
    for i, data4 in enumerate(data_loader,0):
        start = time.time()
        # step1 init
        optimizer.zero_grad()
        # data_loder = input_type, input_value,input_length ,target,parent
        # step 2 prepare the data
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
        loss.backward()
        optimizer.step()
        # loss
        total_loss += loss.item()
        #topv, topi = yt_point.data.topk(1)
        #eval_index = topi.squeeze().detach()
        # print(eval_index)
        end = time.time()
        #print(i, "batch time spend", end - start)
    return total_loss

#4 eval
def eval(model,eval_data,loss_function):
    total_loss = 0
    for i, data4 in enumerate(eval_data,0):
        start = time.time()

        # step1  prepare the data
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
        # loss
        total_loss += loss.item()
        #topv, topi = yt_point.data.topk(1)
        #eval_index = topi.squeeze().detach()
        # print(eval_index)
        end = time.time()
        #print(i, "batch time spend", end - start)
    return total_loss




def main():
    # 3.model init
    model = main_model(len(value_vocab), EMBEDDING_value, len(type_vocab), EMBEDDING_type, HIDDEN_SIZE, BATCH_SIZE,
                       CONTEXT_WINDOW).to(device)
    loss_function = nn.NLLLoss()
    learning_rate = 0.001
    #decay = 0.6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    clip = 5
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    losses_train = []
    losses_eval = []

    staring_training = time.time()
    print("staring training ", staring_training - time_start)

    ##  training
    num_epochs=8
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        train_loss = train(model,optimizer,data_loader_train,loss_function)
        val_loss = eval(model,data_loader_eval,loss_function)
        now = time.time()

        losses_train.append(train_loss / len_train)

        print("[Epoch:%d] train_loss:%.4f val_loss:%5.3f | time spend: %5.2f"
              % (epoch + 1, train_loss / len_train,val_loss/len_eval,(now - time_start)/60))
        losses_eval.append(val_loss / len_eval)


    torch.save(model.state_dict(), 'params_lstm_attn_train_test.pkl')
    model.load_state_dict(torch.load('params_lstm_attn_train_test.pkl'))



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)



