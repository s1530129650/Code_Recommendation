#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: pointer_attn_gpu.py
@time: 2019/4/16 11:00
"""
def dfs_AST(ast_list1, start):
    ast_list =ast_list1
    flatten_ast_list = []
    visited = [0]
    stack = [[start, 0]]

    while stack:
        note_name = "children"
        (v, next_child_idx) = stack[-1]

        if (note_name not in ast_list[v].keys()) or (next_child_idx >= len(ast_list[v]["children"])):

            stack.pop()
            continue
        next_child = ast_list[v]["children"][next_child_idx]
        stack[-1][1] += 1
        if next_child in visited:
            continue
        visited.append(next_child)
        stack.append([next_child, 0])
    #print(visited)
    for i in visited:
       # print(i,"children"  in ast_list[i].keys())

        if "children" in ast_list[i].keys():
            #print(ast_list[i])
            del(ast_list[i]['children'])
        if "value" not in ast_list[i].keys():
                #ast_list[i].pop(note_name)
            ast_list[i]["value"] = "empty"

        flatten_ast_list.append(ast_list[i])
    #print(flatten_ast_list)
    return flatten_ast_list


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random

device = torch.device("cpu")

import time
time_start = time.time()

str =r"D:/v-enshi/Language model/suggestion/Code Completion with Neural Attention and Pointer Networks"


torch.manual_seed(1)

EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5
CONTEXT_WINDOW = 3
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 3
max_vocab_size = 4
'''
EMBEDDING_value = 1200
EMBEDDING_type = 300
HIDDEN_SIZE = 1500
CONTEXT_WINDOW = 50
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 50
max_vocab_size = 1000

# data loading
data = []
with open(filepath,'r') as load_f:
    data1 = load_f.readlines()
for i in range(len(data1)):
    content = json.loads(data1[i])
    data.append(content)

data = [[ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num","value":"1"} ]]

# dfs flatten
'''
def data_loading(filepath):
    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(1):
        content = json.loads(data1[i])
        data.append(content)

    data_flatten = []

    for i  in range(len(data)):
        data_flatten .append(dfs_AST(data[i], 0))
    
    # {type:value} form
    pairs= []
    for datalist in data_flatten:
        inner_data = []
        for items in datalist:
            inner_data.append({items["type"]:items["value"]})
        pairs.append(inner_data)
    return pairs
training_path = str + r"/data/python/python100k_train.json"

training_data = data_loading(training_path)


# build vocabulary
type_to_ix = {"EOF":0}
word_to_ix = {}
for i in range(len(training_data)):
    for it in training_data[i]:
        for type , value in it.items():
            if type not in type_to_ix:
                type_to_ix[type] = len(type_to_ix)
            if value in word_to_ix:
                word_to_ix[value] = word_to_ix[value] + 1
            else:
                word_to_ix[value] = 1


# 1k 10k  50k vocabulary
L = sorted(word_to_ix.items(),key=lambda item:item[1],reverse=True)
value_to_ix = {"UNK":0,"EOF":1}
for i in range(max_vocab_size ):
    value_to_ix [L[i][0]] = len(value_to_ix)

# queries
def Queries(data1):
    data_rd = []
    vocab_size = len(value_to_ix)
    for i in range(len(data1)):
        length = len(data1[i])
        if length <= CONTEXT_WINDOW + 1:
            continue
        rd = random.randint(CONTEXT_WINDOW + 1, len(data1[i]) - 1)  # 取rand int rd
        inner_data = [data1[i][:rd], [data1[i][rd]], -1,
                      rd]  # [context,global position, context position, self -position)
        for j in range(rd - 1, rd - CONTEXT_WINDOW,
                       -1):  # whether the remove node in the context.if the node in context,we remeber the position in context
            if j < 0:
                break  # some sequence length is less than CONTEXT_WINDOW
            if data1[i][rd] == data1[i][j]:
                inner_data = [data1[i][:rd], [data1[i][rd]], vocab_size + rd - j - 1, rd]
                break
        data_rd.append(inner_data)
    return data_rd

training_data_rd = Queries(training_data)
print("training data size:",len(training_data_rd))

class LSTM_component(nn.Module):

    def __init__(self,vocab_value_size, value_dim,vocab_type_size, type_dim,
                 hidden_dim, batch_size):
        super(LSTM_component, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.value_embeddings = nn.Embedding(vocab_value_size, value_dim)
        self.type_embeddings = nn.Embedding(vocab_type_size, type_dim)
        self.lstm = nn.LSTM(value_dim + type_dim, hidden_dim)


    def forward(self,sentence,hc):
        #senten=[[1 2 3],[4  5 6]] sentence[0]is value  index ..[1] is type index
        embeds_value = self.value_embeddings(sentence[0])
        embeds_type  = self.type_embeddings(sentence[1])
        embeds = torch.cat([embeds_value,embeds_type],1).view(len(sentence[0]),1,-1)
        h0 = hc[0]
        c0 = hc[1]
        lstm_out, (lstm_h,lstm_c) = self.lstm(embeds,(h0,c0))

        return  lstm_out ,lstm_h,lstm_c


def prepare_sequence(seq,val_to_ix,ty_to_ix):# trans code to idex
    idxs_ty = []
    idxs_vl = []
    for w in seq:
        for k,v in w.items():
            if v in val_to_ix.keys():
                idxs_vl.append(val_to_ix[v])
            else:
                idxs_vl.append(val_to_ix["UNK"])
            idxs_ty.append(ty_to_ix[k])

    return torch.tensor([idxs_vl ,idxs_ty ],dtype = torch.long,device=device)

class Context_atten(nn.Module):
    def __init__(self, hidden_dim,context_window):
        super(Context_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.Wm = nn.Parameter(torch.ones(hidden_dim,hidden_dim))
        self.V = nn.Parameter(torch.ones(hidden_dim, 1))
        self.linear1 = nn.Linear(hidden_dim,hidden_dim,bias= False)

    def forward(self, inputs, hc):
        #将LSTM那层output的lstm_out ,lstm_h作为context_sttention的input
        #Mt = inputs[-self.context_window - 1:-1, :, :]

        Mt = inputs[-self.context_window-1:-1,:,:]  # context size 实则为少一个，because size of some sequence is less than context size
        Mt = Mt.view(self.context_window, self.hidden_dim) # 3*K
        #Mt = res.transpose(1, 0)
        #At1 = torch.mm(Mt,self.Wm) #(L*k)
        one_TL = torch.ones(context_window,1) #(L,1)
        #linear(hn.view(1,-1)) #(1,k)
        #At2 = torch.mm(one_TL, self.linear1(hn.view(1, -1)))
        # At = vT tanh (Wm*Mt + (Wh*ht)*1TL)
        #V = torch.ones(self.hidden_dim, 1)
        At = torch.mm(torch.tanh(torch.mm(Mt,self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))) , self.V)
        alphat =F.log_softmax(At.view(1,-1),dim = 1) # [1,3]
        ct = torch.mm(alphat,Mt)
        return  alphat,ct



class Parent_atten(nn.Module):
    def __init__(self,hidden_dim,vocab_size):
        super(Parent_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.Wg_linear = nn.Linear(hidden_dim*2,hidden_dim,bias= False)
        self.Wv_linear = nn.Linear(hidden_dim , vocab_size)

    def forward(self,ht,ct):

        #Gt = torch.tanh(self.Wg_linear( torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        Gt = torch.tanh(self.Wg_linear( torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        yt = F.log_softmax(self.Wv_linear(Gt),dim= 1)
        return yt




class Point_Mixture(nn.Module):
    def __init__(self,hidden_dim):
        super(Point_Mixture, self).__init__()
        self.Ws_linear = nn.Linear(hidden_dim*2,1)
    def forward(self, ht,ct,wt,lt):
        # wt是formula 5的 yt,lt是formula(2)的alpha
        st = torch.sigmoid(self.Ws_linear(torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        yt = torch.cat( [torch.mm(st,wt.view(1,-1)) , torch.mm(1 - st,lt.view(1,-1))] ,1)
        #model_par_atten_value = Parent_atten(HIDDEN_SIZE, len(value_to_ix ))

        return yt


class main_model(nn.Module):
    def __init__(self,vocab_value_size, value_dim,vocab_type_size, type_dim,
                 hidden_dim, batch_size,context_window):
        super(main_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.model_LSTM = LSTM_component(vocab_value_size, value_dim, vocab_type_size, type_dim, hidden_dim, batch_size)
        self.model_catten = Context_atten(hidden_dim, context_window)
        self.model_par_atten_value = Parent_atten(hidden_dim, vocab_value_size)
        self.model_Point_Mix = Point_Mixture(hidden_dim)

    def forward(self,inputs,hc):
        output, hn, cn = self.model_LSTM(inputs,hc)

        alpha, out_ct = self.model_catten(output, hn)
        Yt_value = self.model_par_atten_value(hn, out_ct)
        yt_point = self.model_Point_Mix(hn, out_ct, Yt_value, alpha)

        return yt_point

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim,device=device ), torch.zeros(1, self.batch_size, self.hidden_dim,device=device)

## training
model = main_model(len(value_to_ix ),EMBEDDING_value,len(type_to_ix),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW).to(device)

loss_function = nn.NLLLoss()
learning_rate = 0.001
decay = 0.6
optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=decay)
clip = 5
nn.utils.clip_grad_norm_(model.parameters(), clip)

'''
with torch.no_grad():
    inputs = prepare_sequence(training_data_rd[0][0],value_to_ix ,type_to_ix)
    print("inputs",inputs)
    output = model(inputs,model.initHidden())
    print("tag_scorces", output)

'''
losses = []

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure(1)
    #fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
   # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


for epoch in range(8):
    total_loss = 0
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for i in range(len(training_data_rd)):

        sequence = training_data_rd[i][0]
        global_tags = training_data_rd[i][1]
        local_tags  = training_data_rd[i][2]
        targets = prepare_sequence(global_tags, value_to_ix, type_to_ix)
        if local_tags == -1 and targets[0] == 0:
            continue
        optimizer.zero_grad()
        #step 2 prepare the data
        inputs = prepare_sequence(sequence, value_to_ix , type_to_ix)


        if not local_tags == -1 :
            targets[0] = local_tags
        # step1 init

        #step 3 get the scorece


        yt_point = model(inputs, model.initHidden())
        target = targets[0].to(device)
        #step 4 train
        #loss = loss_function(yt_point.view(-1),targets[0])
        loss = loss_function(yt_point.view(1, -1), target)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        topv, topi = yt_point.data.topk(1)
        eval_index = topi.squeeze().detach()

        #print(eval_index)
    now = time.time()
    print('epoch = %d  time spend:%s  loss average%.4f' % (epoch+1,now -time_start,total_loss/len(training_data_rd)))

    losses.append(total_loss/len(training_data_rd))
showPlot(losses)
print(losses)

torch.save(model.state_dict(), 'params1w.pkl')
model.load_state_dict(torch.load('params1w.pkl'))

# envaluate

def evaluate(model):
    with torch.no_grad():
        eval_file = str + "\data\python\python50k_eval.json"
        eval_data = data_loading(eval_file)
        eval_data_rd = Queries(eval_data)
        print("eval data size:", len(eval_data_rd))
        accu = 0
        for i in range(len(eval_data_rd)):
            sequence = eval_data_rd[i][0]
            global_tags = eval_data_rd[i][1]
            local_tags = eval_data_rd[i][2]
            targets = prepare_sequence(global_tags, value_to_ix, type_to_ix)
            if not local_tags == -1:
                targets[0] = local_tags # targets[0] is value targets[1] is type
            #value target
            target = targets[0]
            # input
            inputs = prepare_sequence(sequence, value_to_ix, type_to_ix)
            yt_point = model(inputs, model.initHidden())
            topv, topi = yt_point.data.topk(1)
            eval_index = topi.squeeze().detach().item()
            if  target_index == eval_index :
                accu += 1
        accu = accu /len(eval_data_rd)


