#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: baseLine.py
@time: 2019/4/22 15:40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import numpy as np

device = torch.device("cpu")

import time

time_start = time.time()
torch.manual_seed(1)

EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5
CONTEXT_WINDOW = 5
#BATCH_SIZE = 2
BATCH_SIZE = 1




class LSTM_component(nn.Module):

    def __init__(self, vocab_value_size, value_dim, vocab_type_size, type_dim,
                 hidden_dim, batch_size):
        super(LSTM_component, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.value_embeddings = nn.Embedding(vocab_value_size, value_dim)
        self.type_embeddings = nn.Embedding(vocab_type_size, type_dim)
        self.lstm = nn.LSTM(value_dim + type_dim, hidden_dim)

    def forward(self, sentence, hc):

        embeds_type = self.value_embeddings(sentence[0])
        embeds_value = self.type_embeddings(sentence[1])
        embeds = torch.cat([embeds_value, embeds_type], 1).view(len(sentence[0]), 1, -1)
        h0 = hc[0]
        c0 = hc[1]
        lstm_out, (lstm_h, lstm_c) = self.lstm(embeds, (h0, c0))

        return lstm_out, lstm_h, lstm_c
    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim, device=device), torch.zeros(1, self.batch_size,self.hidden_dim,device=device)

def prepare_sequence(seq, val_to_ix, type_to_ix):  # trans code to idex
    idxs_ty = []
    idxs_vl = []
    for node in seq:
        if "value" in node.keys():
            if node["value"] in val_to_ix.keys():
                idxs_vl.append(val_to_ix[ node["value"] ])
            else:
                idxs_vl.append(val_to_ix["UNK"])
        else:
            idxs_vl.append(val_to_ix["UNK"])
        idxs_ty.append(type_to_ix[node["type"]])
    return torch.tensor([idxs_ty, idxs_vl], dtype=torch.long, device=device)

model_LSTM = LSTM_component(20,EMBEDDING_value,10,EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)

"""
with torch.no_grad():
    inputs =torch.tensor([[ 5, 11,  5, 11,  5, 11,  5, 11,  5, 12],
        [ 0,  0,  0,  0, 6,  0, 8,  0,  0,  7]])
    output, hn,cn = model_LSTM(inputs,model_LSTM.initHidden())
    print("tag_scorces", output, hn,cn)
"""

class Context_atten(nn.Module):
    def __init__(self, hidden_dim,context_window):
        super(Context_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.Wm = nn.Parameter(torch.ones(hidden_dim,hidden_dim))
        self.V = nn.Parameter(torch.ones(hidden_dim, 1))
        self.linear1 = nn.Linear(hidden_dim,hidden_dim,bias= False)

    def forward(self, inputs, hc):

        Mt = inputs[-self.context_window-1:-1,:,:]  # context size 实则为少一个，because size of some sequence is less than context size
        Mt = Mt.view(self.context_window, self.hidden_dim) # 3*K
        one_TL = torch.ones(context_window,1) #(L,1)

        At = torch.mm(torch.tanh(torch.mm(Mt,self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))) , self.V)
        alphat =F.log_softmax(At.view(1,-1),dim = 1) # [1,3]
        ct = torch.mm(alphat,Mt)
        return  alphat,ct