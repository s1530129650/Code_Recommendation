#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: queries.py
@time: 2019/4/19 16:26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random

device = torch.device("cpu")

import time

time_start = time.time()

torch.manual_seed(1)

max_vocab_size = 100
context_window = 100

def data_loading(filepath):
    '''
    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        data.append(content)
    '''
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
        data  = json.loads(data1[0])  #这里之后替换为上面的


    data_flatten = []

    for i in range(len(data)):
        data_flatten.append(dfs_AST(data[i], 0))

    # {type:value} form
    pairs = []
    for datalist in data_flatten:
        inner_data = []
        for items in datalist:
            inner_data.append({items["type"]: items["value"]})
        pairs.append(inner_data)
    return pairs


training_path = r"../data/python/f2.json"

training_data = data_loading(training_path)


print(training_data)
# build vocabulary
type_to_ix = {"EOF": 0}
word_to_ix = {}
for i in range(len(training_data)):
    for it in training_data[i]:
        for type, value in it.items():
            if type not in type_to_ix:
                type_to_ix[type] = len(type_to_ix)
            if value in word_to_ix:
                word_to_ix[value] = word_to_ix[value] + 1
            else:
                word_to_ix[value] = 1

# 1k 10k  50k vocabulary
L = sorted(word_to_ix.items(), key=lambda item: item[1], reverse=True)
value_to_ix = {"UNK": 0, "EOF": 1}
for i in range(max_vocab_size):
    value_to_ix[L[i][0]] = len(value_to_ix)

print(value_to_ix)

# queries
def Queries(data1):
    data_rd = []
    vocab_size = len(value_to_ix)
    for i in range(len(data1)):
        length = len(data1[i])
        if length <= CONTEXT_WINDOW + 1:
            continue
        rd = random.randint(CONTEXT_WINDOW + 1, len(data1[i]) - 1)
        while "empty" in training_data[i][rd].values(): #1.look for leaf node
            rd = rd + 1
        #inner_data = [data1[i][:rd], [data1[i][rd]], -1, rd]  # [context,global position, context position, self -position)
        for j in range(rd - 1, rd - CONTEXT_WINDOW, -1):  # whether the remove node in the context.if the node in context,we remeber the position in context
            if j < 0:
                break  # some sequence length is less than CONTEXT_WINDOW
            if data1[i][rd] == data1[i][j] and data1[i][rd].values():
                inner_data = [data1[i][:rd], [data1[i][rd]], vocab_size + rd - j - 1, rd]
                break
        data_rd.append(inner_data)
    return data_rd

'''
training_data_rd = Queries(training_data)
now = time.time()
print("time spend", now - time_start)
print("training data size:", len(training_data_rd))

'''
