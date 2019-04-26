#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: Queries1.py
@time: 2019/4/23 15:02
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import numpy as np
import time
torch.manual_seed(1)

#use_gpu = False
use_gpu = True

if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
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

arr=np.load(r"../data/python/training.npz")
value_vocab = arr['value_vocab'].item()
type_vocab = arr['type_vocab'].item()


now = time.time()
print("vocabulary loading",now-time_start)

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
#print(training_queries)


#4. text -> idex
def prepare_sequence(seq, val_to_ix, type_to_ix):  # trans code to idex
    idxs_ty = []
    idxs_vl = []
    UNK = 1
    for node in seq:

        '''
        if node["value"] in val_to_ix.keys():
            idxs_vl.append(val_to_ix[ node["value"] ])
        else:
            idxs_vl.append(val_to_ix["UNK"])
        '''
        value_str = node.get('value', 'UNK')
        type_str  =  node.get('type', 'UNK')
        idxs_vl.append(val_to_ix.get(value_str , UNK))
        idxs_ty.append(type_to_ix.get(type_str,UNK))
    #print("np.array([idxs_ty, idxs_vl])",np.array([idxs_ty, idxs_vl]))
    return [idxs_ty, idxs_vl]

input = []
parent = []
target = []

for i in range(len(training_queries)):
    sequence = training_queries[i][0]
    input.append( prepare_sequence(sequence, value_vocab, type_vocab))
    parent.append(training_queries[i][4])
    target.append(training_queries[i][3])

input  = np.array(input)
parent = np.array(parent)
target = np.array(target)




np.savez('../data/python/eval.npz',input_data = input ,parent_data = parent ,target_data = target, \
                                        value_vocab = value_vocab,type_vocab = type_vocab)
now  = time.time()
print("done",now -  time_start)
#print("value_vocab = ",value_vocab)
print(len(value_vocab))
'''
arr=np.load('data_save.npz')
print (arr['input_data'])
print (arr['parent_data'])
print (arr['target_data'])


with codecs.open(r"..\data\python\QUERIES\python_train.json",'w', 'utf-8') as outf:
    for items in training_queries:
        json.dump(items, outf, ensure_ascii=False)
        outf.write('\n')

'''