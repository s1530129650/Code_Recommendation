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
    TopK = [1,5,10]

else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100
    EMBEDDING_value = 2
    EMBEDDING_type = 3
    HIDDEN_SIZE = 5
    BATCH_SIZE = 2
    TopK = [1, 5, 10]

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

with np.load(r"../data/python/vocabulary_50k.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()


with open('../data/python/training_ALL_50k.pickle', 'rb') as f:
    data_train = pickle.load(f)
len_train = data_train.length
print("len_train",len_train)
data_loader_train = DataLoader(data_train, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

with open('../data/python/test2_50k.pickle', 'rb') as f:
    data_eval = pickle.load(f)
len_eval = data_eval.length
print("len_eval",len_eval)
data_loader_eval = DataLoader(data_eval, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

now = time.time()
print("data loading",now-time_start)


##3 train
def train(model,optimizer,data_loader,loss_function):
    model.train()
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
        yt_point = model(input,input_len, parent)
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
    model.eval()
    #AP
    def AP(pre, ground_true):
        Ap = 0
        for i in range(len(pre)):
            if pre[i] == ground_true:
                Ap = Ap + 1 / (i + 1)
        return Ap

    total_loss = 0
    A_P = []
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
        # step 2 get the scorece
        yt_point = model(input,input_len, parent)
        #step 3 MAP
        output = yt_point.squeeze(0)
        #topK = [1, 2, 10]
        _, pred = output.topk(TopK[1], 1)

        for batch in range(BATCH_SIZE):
            A_P.append(AP(pred[batch], target[batch]))
        # step 4 loss
        loss = loss_function(yt_point.squeeze(0), target.to(device))

        # loss
        total_loss += loss.item()
        #topv, topi = yt_point.data.topk(1)
        #eval_index = topi.squeeze().detach()
        # print(eval_index)
        end = time.time()
        #print(i, "batch time spend", end - start)
    m_a_p = sum(A_P) / len_eval
    return total_loss,m_a_p

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
    M_A_P = []

    staring_training = time.time()
    print("staring training ", staring_training - time_start)

    ##  training
    num_epochs= 100
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        train_loss = train(model,optimizer,data_loader_train,loss_function)
        val_loss ,m_A_p = eval(model,data_loader_eval,loss_function)
        now = time.time()

        losses_train.append(train_loss / len_train)

        print("[Epoch:%d] train_loss:%f val_loss:%f | time spend: %f"
              % (epoch + 1, train_loss / len_train,val_loss/len_eval,(now - time_start)/60))
        losses_eval.append(val_loss / len_eval)
        M_A_P.append(m_A_p)
        if epoch % 10 == 0 :
            print("train loss",losses_train)
            print("eval loss",losses_eval)
            print("eval MAP", M_A_P)
        torch.save(model.state_dict(),r"./para/"+str(epoch+1)+'params_lstm_attn_50k.pkl')
    model.load_state_dict(torch.load(r"./para/"+str(epoch+1)+'params_lstm_attn_50k.pkl'))

    import pandas as pd

    dataframe = pd.DataFrame({'train_loss': losses_train, 'eval_loss': losses_eval,"eval MAP": M_A_P})
    dataframe.to_csv("test.csv", index=False, sep=',')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)



