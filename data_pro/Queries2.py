#!/usr/bin/env python
#!-*-coding:utf-8 -*-

"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com@site:
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
import time
torch.manual_seed(1)

import json # load data
import  pickle #save data

use_gpu = False
use_gpu = True

if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 10000
    CONTEXT_WINDOW = 100
    min_snippet_len = 3
else:
    device = torch.device("cpu")
    max_vocab_size = 100
    CONTEXT_WINDOW = 100
    min_snippet_len = 3


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
    training_path = r"../data/python/python100k_train.json"
else:
    str = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
    training_path = str + r"\data\python\f10_.json"


training_data = data_loading(training_path)



now = time.time()
print("data loading",now-time_start)
## 2. build vocabulary
def build_vocab(data):
    type_to_ix = {}
    word_to_ix = {}
    for i in range(len(data)):
        for item in data[i]:
            '''
            if item["type"] not in type_to_ix:
                type_to_ix[item["type"]] = len(type_to_ix)
            '''
            if "type" in item.keys():
                if item["type"] in type_to_ix:
                    type_to_ix[item["type"]] = type_to_ix[item["type"]] + 1
                else:
                    type_to_ix[item["type"]] = 1

            if "value" in item.keys():
                if item["value"] in word_to_ix:
                    word_to_ix[item["value"]] = word_to_ix[item["value"]] + 1
                else:
                    word_to_ix[item["value"]] = 1

    # 1k 10k  50k vocabulary
    L = sorted(word_to_ix.items(), key=lambda item: item[1], reverse=True)
    list1 = list(word_to_ix.values())

    #print("terminal",np.sum(list1))
    value_to_ix = {"EOF": 0,"UNK":1}
    for i in range(max_vocab_size):
        value_to_ix[L[i][0]] = len(value_to_ix)

    L1 = sorted(type_to_ix.items(), key=lambda item: item[1], reverse=True)
    type_to_ix = {"EOF": 0, "UNK": 1}
    for i in range(len(L1)):
        type_to_ix[L[i][0]] = len(type_to_ix)
    #print(L1)
    return type_to_ix, value_to_ix

type_vocab,value_vocab = build_vocab(training_data)


## 4 make queries
def make_queries(dataList):
    queries = []
    for data in dataList:  # one AST
        length = len(data)
        if length < min_snippet_len or length > 2000:  # quit too small sample
            continue

        for i in range(min_snippet_len, length):  # make queries for one node
            if "value" not in data[i]:  # fine terminal
                continue
            # find same terminal  in the context
            for j in range(i - 1, max(i - CONTEXT_WINDOW - 1, -1), -1):  #
                if "value" in data[j] and data[j]["value"] == data[i]["value"]:
                    Tree = data[:i]

                    queries.append(Tree)
                    break
                    # print(Tree,label)
        print("len",len(queries))

    return queries

quer_data = make_queries(training_data)
print(len(quer_data))

#print(len(type_vocab))
#print(len(value_vocab))
#print(type_vocab)
#js = json.dumps(type_vocab)
"""
with open('type.txt', 'w') as f:
    f.write(js)


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

        value_str = node.get('value', 'UNK')
        idxs_vl.append(val_to_ix.get(value_str, UNK))
        idxs_ty.append(type_to_ix[node.get('type', 'UNK')])
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
#print("input",input)
input  = np.array(input)
parent = np.array(parent)
target = np.array(target)




np.savez('../data/python/training.npz',input_data = input ,parent_data = parent ,target_data = target, \
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
"""