#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: model3.py
@time: 2019/4/23 22:38
epoch = 16 , add testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import numpy as np
import time


time_start = time.time()
torch.manual_seed(1)

use_gpu = False
#use_gpu = True
if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100

##1. data loading
arr=np.load(r"../data/python/training.npz")
inputs = arr['input_data']
parents = arr['parent_data']
targets = arr['target_data']
value_vocab = arr['value_vocab'].item()
type_vocab = arr['type_vocab'].item()

arr = np.load(r"../data/python/eval.npz")
inputs_test = arr['input_data']
parents_test = arr['parent_data']
targets_test = arr['target_data']








now = time.time()
print("data loading", now - time_start)


##2. parameters setting
if use_gpu:
    EMBEDDING_value = 1200
    EMBEDDING_type = 300
    HIDDEN_SIZE = 1500
    BATCH_SIZE = 1
else:
    EMBEDDING_value = 2
    EMBEDDING_type = 3
    HIDDEN_SIZE = 5

    BATCH_SIZE = 1






## 3.1 LSTM component
class LSTM_component(nn.Module):

    def __init__(self, vocab_value_size, value_dim, vocab_type_size, type_dim,
                 hidden_dim, batch_size, context_window, dropout_p=0.5):
        super(LSTM_component, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.context_window = context_window
        self.value_embeddings = nn.Embedding(vocab_value_size, value_dim)
        self.type_embeddings = nn.Embedding(vocab_type_size, type_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(value_dim + type_dim, hidden_dim)

    def forward(self, sentence, hc):
        embeds_type = self.type_embeddings(sentence[0])
        embeds_value = self.value_embeddings(sentence[1])
        embeds = torch.cat([embeds_value, embeds_type], 1).view(len(sentence[0]), 1, -1)
        h0 = hc[0]
        c0 = hc[1]
        embeds[:-self.context_window]

        lstm_out1, (lstm_h1, lstm_c1) = self.lstm(self.dropout(embeds[:-self.context_window]), (h0, c0))

        lstm_out, (lstm_h, lstm_c) = self.lstm(embeds[-self.context_window:], (lstm_h1, lstm_c1))

        return lstm_out, lstm_h, lstm_c

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim, device=device), torch.zeros(1, self.batch_size,
                                                                                            self.hidden_dim,
                                                                                            device=device)

'''
model_LSTM = LSTM_component(20,EMBEDDING_value,10,EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)


with torch.no_grad():
    inputs =torch.tensor([[ 5, 11,  5, 11,  5, 11,  5, 11,  5, 12],
        [ 0,  0,  0,  0, 6,  0, 8,  0,  0,  7]])
    output, hn,cn = model_LSTM(inputs,model_LSTM.initHidden())
    print("tag_scorces", output, hn,cn)
'''

## 3.2 attention component
class Context_atten(nn.Module):
    def __init__(self, hidden_dim, context_window, dropout_p=0.25):
        super(Context_atten, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window

        self.Wm = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.V = nn.Parameter(torch.ones(hidden_dim, 1))
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, hc):
        # Mt = inputs[-self.context_window:,:,:]
        Mt = inputs.view(self.context_window, self.hidden_dim)  #
        one_TL = torch.ones(self.context_window, 1, device=device)  # (L,1)

        At = torch.mm(torch.tanh(torch.mm(Mt, self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))), self.V)
        alphat = F.log_softmax(At.view(1, -1), dim=1)  # [1,3]
        ct = torch.mm(alphat, Mt)
        return alphat, ct


'''
model_catten = Context_atten(HIDDEN_SIZE, CONTEXT_WINDOW)

with torch.no_grad():
    alpha , out_ct = model_catten (output, hn)
    print(out_ct)
'''


## 3.3parent attention
class Parent_atten(nn.Module):
    def __init__(self, hidden_dim, context_window):
        super(Parent_atten, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.Wg_linear = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.Wv_linear = nn.Linear(hidden_dim, self.context_window)

    def forward(self, ht, ct, pt):
        Gt = torch.tanh(self.Wg_linear(torch.cat([ht.view(1, -1), ct.view(1, -1), pt.view(1, -1)], 1)))
        yt = F.log_softmax(self.Wv_linear(Gt), dim=1)
        return yt


'''
model_par_atten_type = Parent_atten(HIDDEN_SIZE, CONTEXT_WINDOW)
with torch.no_grad():
    Yt_type = model_par_atten_type  ( hn,out_ct,output[6])
    print(Yt_type)

'''

## 3 main model
class main_model(nn.Module):
    def __init__(self, vocab_value_size, value_dim, vocab_type_size, type_dim,
                 hidden_dim, batch_size, context_window, dropout_p=0.25):
        super(main_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.context_window = context_window

        self.model_LSTM = LSTM_component(vocab_value_size, value_dim, vocab_type_size, type_dim, hidden_dim, batch_size,
                                         context_window).to(device)
        self.model_catten = Context_atten(hidden_dim, context_window).to(device)
        self.model_par_atten_value = Parent_atten(hidden_dim, vocab_value_size).to(device)

    def forward(self, inputs, hc, parent):
        output, hn, cn = self.model_LSTM(inputs, hc)
        alpha, out_ct = self.model_catten(output, hn)
        Yt = self.model_par_atten_value(hn, out_ct, output[-parent - 1])
        return Yt

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim, device=device), torch.zeros(1, self.batch_size,
                                                                                            self.hidden_dim,
                                                                                            device=device)


## 4 training
model = main_model(len(value_vocab), EMBEDDING_value, len(type_vocab), EMBEDDING_type, HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW).to(device)
loss_function = nn.NLLLoss()
learning_rate = 0.01
decay = 0.6
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay)
clip = 5
nn.utils.clip_grad_norm_(model.parameters(), clip)

losses = []
eval_losses = []
length = len(targets)
for epoch in range(8):
    total_loss = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # query = [context,   predict_node,    position,   same_node_position,   parent_node_position]
    #print(len(targets))
    for i in range(len(targets)):

        # step1 init
        optimizer.zero_grad()
        # step 2 prepare the data
        #print("inputs[i]",i,inputs[i])
        input = [torch.tensor(inputs[i][0] ,device=device),torch.tensor(inputs[i][1],device=device)]
        parent = parents[i]
        target = torch.tensor([targets[i]], dtype =torch.long,device=device)
        #

        # step 3 get the scorece
        yt_point = model(input, model.initHidden(), parent)
        #print("y_point",yt_point)

        # step 4 train
        loss = loss_function(yt_point.view(1, -1), target)

        loss.backward()
        optimizer.step()

        # loss
        total_loss += loss.item()
        #topv, topi = yt_point.data.topk(1)
        #eval_index = topi.squeeze().detach()
        # print(eval_index)
    now = time.time()
    print('epoch = %d  time spend:%s  loss average%.4f' % (
    epoch + 1, now - time_start, total_loss / length))


    losses.append(total_loss / length)

print(losses)
torch.save(model.state_dict(), 'params_lstm_attn.pkl')
model.load_state_dict(torch.load('params_lstm_attn.pkl'))
'''
# 5 testing
with torch.no_grad():
    for i in range(len(targets_test)):
        # step 2 prepare the data
        input = [torch.tensor(inputs_test[i][0], device=device), torch.tensor(inputs_test[i][1], device=device)]
        parent = parents_test[i]
        target = torch.tensor([targets_test[i]], dtype=torch.long, device=device)
        # step 3 get the scorece
        yt_point = model(input, model.initHidden(), parent)
        # step 4 train
        loss = loss_function(yt_point.view(1, -1), target)
'''

