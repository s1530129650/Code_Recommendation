#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence
@contact: 123@qq.com
@site:
@software: PyCharm
@file: Queries2.py
@time: 2019/4/22 14:21
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

device = torch.device("cpu")

import time

time_start = time.time()
torch.manual_seed(1)
max_vocab_size = 100
CONTEXT_WINDOW = 100


##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
def data_loading(filepath):

    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        data.append(content)
    return data


str = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
training_path = str + r"\data\python\f2_.json"
training_data = data_loading(training_path)


## 2. build vocabulary
def build_vocab(data):
    type_to_ix = {"EOF": 0}
    word_to_ix = {}
    for i in range(len(data)):
        for item in data[i]:
            if item["type"] not in type_to_ix:
                type_to_ix[item["type"]] = len(type_to_ix)
            if "value" in item.keys():
                if item["value"] in word_to_ix:
                    word_to_ix[item["value"]] = word_to_ix[item["value"]] + 1
                else:
                    word_to_ix[item["value"]] = 1

    # 1k 10k  50k vocabulary
    L = sorted(word_to_ix.items(), key=lambda item: item[1], reverse=True)
    value_to_ix = {"UNK": 0, "EOF": 1}
    for i in range(max_vocab_size):
        value_to_ix[L[i][0]] = len(value_to_ix)
    return type_to_ix, value_to_ix

type_vocab,value_vocab = build_vocab(training_data)

# 3. make the queries
def Queries(data):
    data_rd = []
    random = np.random.RandomState(0)
    for i in range(len(data)):
        length = len(data[i])
        if length <= CONTEXT_WINDOW + 1:
            continue
        rd = random.randint(CONTEXT_WINDOW, length - 1)
        while "value" not in data[i][rd].keys():  # 1.look for leaf node
            rd = rd + 1
            if rd >= length:
                break
        if rd >= length:
            continue

        query = []
        # find same node in the context
        for j in range(rd - 1, rd - CONTEXT_WINDOW - 1,
                       -1):  # whether the remove node in the context.if the node in context,we remeber the position in context

            if data[i][rd]["type"] == data[i][j]["type"] and "value" in data[i][j].keys() and data[i][rd]["value"] == \
                    data[i][j]["value"]:
                #print("j$$$$$$$$$$$",rd - 1, rd - CONTEXT_WINDOW - 1,j,rd - j - 1)
                query = [data[i][:rd], [data[i][rd]], rd, rd - j - 1]
                break
        if j == rd - CONTEXT_WINDOW:  # there is no same node in context
            continue
        # add parents node
        for j in range(rd - 1, rd - CONTEXT_WINDOW - 1, -1):

            if "children" in data[i][j].keys() and rd in data[i][j]["children"]:
                query.append(rd - j - 1)
                break
            if j == rd - CONTEXT_WINDOW:
                query.append(CONTEXT_WINDOW - 1)
                break
        # query = [context,predict_node,position, same_node_position,parent_node_position]
        data_rd.append(query)
    return data_rd

training_queries = Queries(training_data)
#print(training_queries)
##

EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5

BATCH_SIZE = 1

'''
EMBEDDING_value = 1200
EMBEDDING_type = 300
HIDDEN_SIZE = 1500
CONTEXT_WINDOW = 50
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 50
max_vocab_size = 1000
'''
class LSTM_component(nn.Module):

    def __init__(self, vocab_value_size, value_dim, vocab_type_size, type_dim,
                 hidden_dim, batch_size,context_window,dropout_p=0.5):
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

        lstm_out, (lstm_h, lstm_c) = self.lstm(self.dropout(embeds[:-self.context_window]), (h0, c0))
        lstm_out, (lstm_h, lstm_c) = self.lstm(embeds[-self.context_window:], (lstm_h, lstm_c))
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
'''
model_LSTM = LSTM_component(20,EMBEDDING_value,10,EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)


with torch.no_grad():
    inputs =torch.tensor([[ 5, 11,  5, 11,  5, 11,  5, 11,  5, 12],
        [ 0,  0,  0,  0, 6,  0, 8,  0,  0,  7]])
    output, hn,cn = model_LSTM(inputs,model_LSTM.initHidden())
    print("tag_scorces", output, hn,cn)
'''

class Context_atten(nn.Module):
    def __init__(self, hidden_dim,context_window,dropout_p=0.25):
        super(Context_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window



        self.Wm = nn.Parameter(torch.ones(hidden_dim,hidden_dim))
        self.V = nn.Parameter(torch.ones(hidden_dim, 1))
        self.linear1 = nn.Linear(hidden_dim,hidden_dim,bias= False)

    def forward(self, inputs, hc):

        #Mt = inputs[-self.context_window:,:,:]
        Mt = inputs.view(self.context_window, self.hidden_dim) #
        one_TL = torch.ones(self.context_window,1,device = device) #(L,1)

        At = torch.mm(torch.tanh(torch.mm(Mt,self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))) , self.V)
        alphat =F.log_softmax(At.view(1,-1),dim = 1) # [1,3]
        ct = torch.mm(alphat,Mt)
        return  alphat,ct
'''
model_catten = Context_atten(HIDDEN_SIZE, CONTEXT_WINDOW)

with torch.no_grad():
    alpha , out_ct = model_catten (output, hn)
    print(out_ct)
'''

class Parent_atten(nn.Module):
    def __init__(self,hidden_dim,context_window):
        super(Parent_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.Wg_linear = nn.Linear(hidden_dim*3,hidden_dim,bias= False)
        self.Wv_linear = nn.Linear(hidden_dim , self.context_window)

    def forward(self,ht,ct,pt):


        Gt = torch.tanh(self.Wg_linear( torch.cat([ht.view(1,-1),ct.view(1,-1),pt.view(1,-1)],1)))
        yt = F.log_softmax(self.Wv_linear(Gt),dim= 1)
        return yt
'''
model_par_atten_type = Parent_atten(HIDDEN_SIZE, CONTEXT_WINDOW)
with torch.no_grad():
    Yt_type = model_par_atten_type  ( hn,out_ct,output[6])
    print(Yt_type)
    
'''


class main_model(nn.Module):
    def __init__(self,vocab_value_size, value_dim,vocab_type_size, type_dim,
                 hidden_dim, batch_size,context_window,dropout_p=0.25):
        super(main_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.context_window = context_window

        self.model_LSTM = LSTM_component(vocab_value_size, value_dim, vocab_type_size, type_dim, hidden_dim, batch_size,context_window).to(device)
        self.model_catten = Context_atten(hidden_dim, context_window).to(device)
        self.model_par_atten_value = Parent_atten(hidden_dim, context_window).to(device)


    def forward(self,inputs,hc,parent):
        output, hn, cn = self.model_LSTM(inputs,hc)
        #print(output.shape)
        alpha, out_ct = self.model_catten(output, hn)
        Yt = self.model_par_atten_value(hn, out_ct,output[-parent-1])
        return Yt

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim,device=device ), torch.zeros(1, self.batch_size, self.hidden_dim,device=device)

## training
model = main_model(len(value_vocab ),EMBEDDING_value,len(type_vocab),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW).to(device)

loss_function = nn.NLLLoss()
learning_rate = 0.01
decay = 0.6
optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=decay)
clip = 5
nn.utils.clip_grad_norm_(model.parameters(), clip)


losses = []

for epoch in range(8):
    total_loss = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # query = [context,   predict_node,    position,   same_node_position,   parent_node_position]
    for i in range(len(training_queries)):
        sequence = training_queries[i][0]
        inputs = prepare_sequence(sequence, value_vocab, type_vocab)
        # step1 init
        optimizer.zero_grad()
        #step 2 prepare the data
        inputs = prepare_sequence(sequence, value_vocab, type_vocab)

        #step 3 get the scorece
        parent = training_queries[i][4]
        yt_point = model(inputs, model.initHidden(),parent)

        target = torch.tensor([training_queries[i][3]],dtype = torch.long,device = device)

        #step 4 train
        loss = loss_function(yt_point.view(1, -1), target)

        loss.backward()
        optimizer.step()

        #loss
        total_loss += loss.item()
        topv, topi = yt_point.data.topk(1)
        eval_index = topi.squeeze().detach()
        #print(eval_index)
    now = time.time()
    print('epoch = %d  time spend:%s  loss average%.4f' % (epoch+1,now -time_start,total_loss/len(training_queries)))

    losses.append(total_loss/len(training_queries))

print(losses)
torch.save(model.state_dict(), 'params_lstm_attn.pkl')
model.load_state_dict(torch.load('params_lstm_attn.pkl'))