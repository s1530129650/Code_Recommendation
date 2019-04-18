#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: main.py
@time: 2019/3/18 15:15
"""
# pytorch 实现lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dfs import dfs_AST

torch.manual_seed(1)

EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5
context_window = 3
BATCH_SIZE = 1

data = [ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num","value":"1"} ]

data_flatten = dfs_AST(data, 0)

print("data_flatten:",data_flatten)

training_data= []
for items in data_flatten:
    training_data.append({items["type"]:items["value"]})

print("training_data:",training_data)

if not len(training_data) % context_window == 0:
    pad = {"EOF":"EOF"}
    for i in range(context_window - len(training_data) % context_window):
        training_data.append(pad)

type_to_ix = {"EOF":0}
word_to_ix = {"UNK":0,"EOF":1,"empty":2}

for it in training_data:
    for type , value in it.items():
        if type not in type_to_ix:
            type_to_ix[type] = len(type_to_ix)
        if value not in word_to_ix:
            word_to_ix[value] = len(word_to_ix)


print("word_to_ix",word_to_ix)
print("type_to_ix",type_to_ix)




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

model = LSTM_component(len(word_to_ix),EMBEDDING_value,len(type_to_ix),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.1)


def prepare_sequence(seq,val_to_ix,ty_to_ix):
    idxs_ty = []
    idxs_vl = []
    for w in seq:
        for k,v in w.items():
            idxs_ty.append(ty_to_ix[k])
            idxs_vl.append(val_to_ix[v])
    return torch.tensor([idxs_vl ,idxs_ty ],dtype = torch.long)

with torch.no_grad():
    inputs = prepare_sequence(training_data,word_to_ix,type_to_ix)
    print(inputs)
    output, hn,cn = model(inputs,model.initHidden())
    print("tag_scorces", output, hn,cn)


class Context_atten(nn.Module):
    def __init__(self, hidden_dim,context_window):
        super(Context_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.coefficient = torch.nn.Parameter(torch.Tensor([1.55]))

