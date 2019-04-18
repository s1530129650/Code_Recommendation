#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: main2.py
@time: 2019/3/20 21:17
"""
#pytorch 实现 LSTM ATTENTION Pointer等NN
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
max_vocab_size = 100

data = [[ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num","value":"1"} ]]
'''
data = []
with open(str+"\data\python\python100k_train.json",'r') as load_f:
    data1 = load_f.readlines()
for i in range(len(data1)):
    content = json.loads(data1[i])
    data.append(content)
'''
data_flatten = []
for i  in range(len(data)):
    data_flatten .append(dfs_AST(data[i], 0))

#print("data_flatten:",data_flatten)

training_data= []
for datalist in data_flatten:
    inner_data = []
    for items in datalist:
        inner_data.append({items["type"]:items["value"]})
    training_data.append(inner_data)
#print("training_data:",training_data)

type_to_ix = {"EOF":0}
word_to_ix = {"UNK":0,"EOF":1,"empty":2}
for i in range(len(training_data)):
    for it in training_data[i]:
        for type , value in it.items():
            if type not in type_to_ix:
                type_to_ix[type] = len(type_to_ix)
            if value not in word_to_ix:
                word_to_ix[value] = len(word_to_ix)


#print("word_to_ix",word_to_ix)
#print("type_to_ix",type_to_ix)

training_data_rd = []
vocab_size = len(word_to_ix)
for i in range(len(training_data)):
    rd = random.randint(4, len(training_data[i]) - 1)
    flag = False
    while "empty" in training_data[i][rd].values():
        rd = rd + 1
    inner_data = [training_data[i][:rd], [training_data[i][rd]], -1, rd]
    for j in range(rd - CONTEXT_WINDOW, rd):
        if training_data[i][rd] == training_data[i][j]:
            inner_data = [training_data[i][:rd], [training_data[i][rd]], vocab_size+rd - j -1, rd]
            break
    training_data_rd.append(inner_data)
#print(training_data_rd)


for i in range(len(training_data_rd)):
    if not len(training_data_rd[i]) % context_window == 0:
        pad = {"EOF":"EOF"}
        for j in range(context_window - len(training_data_rd[i]) % context_window):
            training_data_rd[i].append(pad)
#print("training_data:",training_data_rd)

#print(training_data_rd)

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

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim ), torch.zeros(1, self.batch_size, self.hidden_dim)

model_LSTM = LSTM_component(len(word_to_ix),EMBEDDING_value,len(type_to_ix),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer_LSTM = optim.SGD(model_LSTM.parameters(),lr = 0.1)


def prepare_sequence(seq,val_to_ix,ty_to_ix):
    idxs_ty = []
    idxs_vl = []
    for w in seq:
        for k,v in w.items():
            idxs_ty.append(ty_to_ix[k])
            idxs_vl.append(val_to_ix[v])
    return torch.tensor([idxs_vl ,idxs_ty ],dtype = torch.long)
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data_rd[0][0],word_to_ix,type_to_ix)
    print(inputs)
    output, hn,cn = model_LSTM(inputs,model_LSTM.initHidden())
    print("tag_scorces", output, hn,cn)
'''

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
        Mt = inputs[-self.context_window - 1:-1, :, :]
        Mt = Mt.view(self.context_window, self.hidden_dim) # 3*K
        #Mt = res.transpose(1, 0)
        #At1 = torch.mm(Mt,self.Wm) #(L*k)
        one_TL = torch.ones(context_window,1) #(L,1)
        #linear(hn.view(1,-1)) #(1,k)
        #At2 = torch.mm(one_TL, self.linear1(hn.view(1, -1)))
        # At = vT tanh (Wm*Mt + (Wh*ht)*1TL)
        #V = torch.ones(self.hidden_dim, 1)
        print("begin")
        At = torch.mm(torch.tanh(torch.mm(Mt,self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))) , V)
        print("end")
        alphat =F.log_softmax(At.view(1,-1),dim = 1) # [1,3]
        ct = torch.mm(alphat,Mt)
        return  alphat,ct


model_catten = Context_atten(HIDDEN_SIZE, CONTEXT_WINDOW)

'''
with torch.no_grad():
    alpha , out_ct = model_catten (output, hn)
    print(out_ct)  
'''

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

model_par_atten_type = Parent_atten(HIDDEN_SIZE, len(type_to_ix))
'''
with torch.no_grad():
    Yt_type = model_par_atten_type  ( hn,out_ct)
    print(Yt_type)
'''


model_par_atten_value = Parent_atten(HIDDEN_SIZE, len(word_to_ix))

'''
with torch.no_grad():
    Yt_value = model_par_atten_value ( hn,out_ct)
    print(Yt_value)
'''


class Point_Mixture(nn.Module):
    def __init__(self,hidden_dim):
        super(Point_Mixture, self).__init__()
        self.Ws_linear = nn.Linear(hidden_dim*2,1)
    def forward(self, ht,ct,wt,lt):
        # wt是formula 5的 yt,lt是formula(2)的alpha
        st = torch.sigmoid(self.Ws_linear(torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        yt = torch.cat( [torch.mm(st,wt.view(1,-1)) , torch.mm(1 - st,lt.view(1,-1))] ,1)

        return yt


model_Point_Mix = Point_Mixture(HIDDEN_SIZE)
'''
with torch.no_grad():
    yt_point = model_Point_Mix(hn,out_ct, Yt_value,alpha )
    print(yt_point)

'''
#loss_function = nn.NLLLoss()

learning_rate = 0.001
decay = 0.6
LSTM_optimizer = optim.SGD(model_LSTM.parameters(), lr=learning_rate,weight_decay=decay)
context_atten_optimizer = optim.SGD(model_catten.parameters(), lr=learning_rate,weight_decay=decay)
parent_atten_optimizer = optim.SGD(model_par_atten_value.parameters(), lr=learning_rate,weight_decay=decay)
point_optimizer = optim.SGD(model_Point_Mix.parameters(), lr=learning_rate,weight_decay=decay)


clip = 5
nn.utils.clip_grad_norm_(model_LSTM.parameters(), clip)
nn.utils.clip_grad_norm_(model_catten.parameters(), clip)
nn.utils.clip_grad_norm_(model_par_atten_value.parameters(), clip)
nn.utils.clip_grad_norm_(model_Point_Mix.parameters(), clip)


losses = []
for epoch in range(8):
    total_loss = 0
    for i in range(len(training_data_rd)):

        sequence = training_data_rd[i][0]
        global_tags = training_data_rd[i][1]
        local_tags  = training_data_rd[i][2]

        LSTM_optimizer.zero_grad()
        context_atten_optimizer.zero_grad()
        parent_atten_optimizer.zero_grad()
        point_optimizer.zero_grad()



        #step 2 prepare the data
        inputs = prepare_sequence(sequence, word_to_ix, type_to_ix)
        targets = prepare_sequence(global_tags, word_to_ix, type_to_ix)
        if not local_tags == -1 :
            targets[0] = local_tags
        # step1 init


        #step 3 get the scorece
        output, hn, cn = model_LSTM(inputs, model_LSTM.initHidden())
        alpha, out_ct = model_catten(output, hn)
        Yt_value = model_par_atten_value(hn, out_ct)
        #Yt_type = model_par_atten_type(hn, out_ct)
        yt_point = model_Point_Mix(hn, out_ct, Yt_value, alpha)


        #step 4 train
        #loss = loss_function(yt_point.view(-1),targets[0])
        loss = loss_function(yt_point.view(1, -1), targets[0])
        loss.backward()

        LSTM_optimizer.step()
        context_atten_optimizer.step()
        parent_atten_optimizer.step()
        point_optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)




time_end=time.time()

print('totally cost',time_end-time_start)