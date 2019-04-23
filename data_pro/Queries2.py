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
import json
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



#4. text -> idex
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

input = np.array([])
parent = np.array([])
target = np.array([])



for i in range(len(training_queries)):
    sequence = training_queries[i][0]
    input  = np.append(input, prepare_sequence(sequence, value_vocab, type_vocab))
    parent = np.append(parent,training_queries[i][4])
    target = np.append(target,torch.tensor([training_queries[i][3]], dtype=torch.long, device=device))


np.savez('array_save.npz',input_data = input ,parent_data = parent ,target_data = target)

arr=np.load('array_save.npz')
print (arr['input_data'])
print (arr['parent_data'])
print (arr['target_data'])

"""

with codecs.open(r"..\data\python\QUERIES\python_train.json",'w', 'utf-8') as outf:
    for items in training_queries:
        json.dump(items, outf, ensure_ascii=False)
        outf.write('\n')

"""