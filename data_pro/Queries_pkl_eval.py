#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: Queries_pkl_eval.py
@time: 2019/4/26 11:27
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
@file: Queries_pkl.py
@time: 2019/4/25 9:46
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  json
import random
import numpy as np
import time
torch.manual_seed(1)

use_gpu = False
use_gpu = True

if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 50000
    CONTEXT_WINDOW = 100
else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100


time_start = time.time()

##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
def data_loading(filepath):

    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        data.append(content)

    return data

if use_gpu:
    training_path = r"../data/python/python50k_eval.json"
else:
    str = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
    training_path = str + r"\data\python\f10_.json"


training_data = data_loading(training_path)

now = time.time()
print("data loading",now-time_start)
## 2. load vocabulary
with np.load(r"../data/python/vocabulary_50k.npz", allow_pickle=True) as arr:
    value_vocab = arr['value_vocab'].item()
    type_vocab = arr['type_vocab'].item()



now = time.time()
print("build vocabulary",now-time_start)

# 3. make the queries
def Queries(data):
    data_rd = []
    random = np.random.RandomState(0)
    for i in range(len(data)):
        length = len(data[i])
        if length <= CONTEXT_WINDOW + 2:
            continue
        rd = random.randint(CONTEXT_WINDOW + 1, length - 1)
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
training_queries.sort( key=lambda x: x[2],reverse=True) # sort
#print(training_queries)


now = time.time()
print("make the queries",now-time_start)


#4 text -> index
def prepare_sequence(seq, val_to_ix, type_to_ix):  # trans code to idex
    idxs_ty = []
    idxs_vl = []
    UNK = 1
    for node in seq:

        value_str = node.get('value', 'UNK')
        idxs_vl.append(val_to_ix.get(value_str, UNK))
        idxs_ty.append(type_to_ix.get(node.get('type', 'UNK'),UNK))
    #print("np.array([idxs_ty, idxs_vl])",np.array([idxs_ty, idxs_vl]))
    return torch.tensor([idxs_vl, idxs_ty],dtype = torch.long)

input_value = []
input_type = []
parent = []
target = []

for i in range(len(training_queries)):
    sequence = training_queries[i][0]
    [input_val, input_ty] = prepare_sequence(sequence, value_vocab, type_vocab)
    par = torch.tensor(training_queries[i][4],dtype = torch.long)
    targ = torch.tensor(training_queries[i][3],dtype = torch.long)

    input_value.append(input_val)
    input_type.append(input_ty)
    parent.append(par)
    target.append(targ)

now = time.time()
print("text -> index",now-time_start)


#5 padding and save
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data

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


x = rnn_utils.pad_sequence(input_value, batch_first=True)
y = rnn_utils.pad_sequence(input_type, batch_first=True)
dataAll = MyData(input_value,x,y,target,parent)
#print(dataAll.data_length)

now = time.time()
print("5. padding ",now-time_start)

# save
import pickle
with open('../data/python/eval_50k.pickle', 'wb') as f:
    pickle.dump(dataAll, f, protocol=pickle.HIGHEST_PROTOCOL)








