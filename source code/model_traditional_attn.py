#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: model_traditional_attn.py
@time: 2019/4/26 17:51
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

import random
import numpy as np

torch.manual_seed(1)
use_gpu = False
#use_gpu = True

##1. parameters setting
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


## 3.1 LSTM component
class LSTM_component(nn.Module):

    def __init__(self, vocab_value_size, value_dim, vocab_type_size, type_dim,
                 hidden_dim, batch_size, context_window, dropout_p=0.5):
        super(LSTM_component, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.context_window = context_window
        self.value_embeddings = nn.Embedding(vocab_value_size, value_dim, padding_idx=0)

        self.type_embeddings = nn.Embedding(vocab_type_size, type_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(value_dim + type_dim, hidden_dim, batch_first=True)

    def forward(self, sentence, setence_len, hc):
        embeds_type = self.type_embeddings(sentence[0])

        embeds_value = self.value_embeddings(sentence[1])
        embeds = torch.cat([embeds_value, embeds_type], 2)
        h0 = hc[0]
        c0 = hc[1]

        embeded = rnn_utils.pack_padded_sequence(embeds, setence_len, batch_first=True)

        lstm_out, (lstm_h, lstm_c) = self.lstm(embeded, (h0, c0))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        return lstm_out, lstm_h, lstm_c

    def initHidden(self):
        return torch.randn(1, self.batch_size, self.hidden_dim, device=device), torch.randn(1, self.batch_size,
                                                                                            self.hidden_dim,
                                                                                            device=device)


'''

#model_LSTM = LSTM_component(20,EMBEDDING_value,10,EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE, CONTEXT_WINDOW)

model_LSTM = LSTM_component(20,2,20,3,5, 2, 3)
with torch.no_grad():
    inputs = [torch.tensor([[6, 6, 5, 3], [7, 4, 0, 0]]), torch.tensor([[3, 3, 2, 1], [4, 1, 0, 0]])]
    output, hn,cn = model_LSTM(inputs,[4,2],model_LSTM.initHidden())
    #print("tag_scorces", output)
'''


## 3.2 attention component
class Context_atten(nn.Module):
    def __init__(self, hidden_dim, context_window, dropout_p=0.25):
        super(Context_atten, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window

        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, setence_len, hc, parent):
        batch_size = inputs.size(0)
        timestep = self.context_window

        Mt = inputs[0, setence_len[0] - timestep: setence_len[0], :].unsqueeze(0)
        par = inputs[0, setence_len[0] - parent[0] - 1, :].unsqueeze(0)

        for i in range(batch_size - 1):
            Mt = torch.cat([Mt, inputs[i + 1, setence_len[i + 1] - timestep: setence_len[i + 1], :].unsqueeze(0)],
                           0)  # [B,T,H]
            par = torch.cat([par, inputs[i + 1, setence_len[i + 1] - parent[i + 1] - 1, :].unsqueeze(0)], 0)  # [B,H]

        h = hc.repeat(timestep, 1, 1).transpose(0, 1)  # hc[1,B,H] -> [B,T,H]
        attn_weights = self.score(h, Mt).unsqueeze(1)  # [B*T*1]
        context = attn_weights.bmm(Mt)  # (B,1,H)
        return attn_weights, context, par

    def score(self, hidden, attn_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, attn_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(attn_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T] a
        energy = F.softmax(energy, dim=2)
        return energy.squeeze(1)  # [B*T]


'''
    def score(self, hidden, attn_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, attn_outputs], 2)),dim = 2)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(attn_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T] a
        return energy.squeeze(1)  # [B*T]

'''

'''
HIDDEN_SIZE = 5
CONTEXT_WINDOW = 2
model_catten = Context_atten(HIDDEN_SIZE, CONTEXT_WINDOW)

with torch.no_grad():
    alpha , out_ct,parent = model_catten (output,[3,2], hn,[0,1])
    #print(out_ct)

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
        # ht[1,B,H] ct[B,1,H] pt[B,H]
        ct = ct.transpose(0, 1)  # (1,B,H)
        pt = pt.unsqueeze(0)

        Gt = torch.tanh(self.Wg_linear(torch.cat([ht, ct, pt], 2)))
        yt = F.log_softmax(self.Wv_linear(Gt), dim=1)
        return yt


'''
model_par_atten_type = Parent_atten(HIDDEN_SIZE, CONTEXT_WINDOW)
with torch.no_grad():
    Yt_type = model_par_atten_type  ( hn,out_ct,parent)
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

    def forward(self, inputs, input_len, hc, parent):
        output, hn, cn = self.model_LSTM(inputs, input_len, hc)
        alpha, out_ct, par = self.model_catten(output, input_len, hn, parent)
        Yt = self.model_par_atten_value(hn, out_ct, par)
        return Yt

    def initHidden(self):
        # return torch.zeros(1, self.batch_size, self.hidden_dim, device=device), torch.zeros(1, self.batch_size,self.hidden_dim, device=device)
        return torch.randn(1, self.batch_size, self.hidden_dim, device=device), torch.randn(1, self.batch_size,
                                                                                            self.hidden_dim,
                                                                                            device=device)