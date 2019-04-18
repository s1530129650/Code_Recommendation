#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: evaluate.py
@time: 2019/4/16 10:58
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
time_start=time.time()

str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"


torch.manual_seed(1)

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
with open(str+"\data\python\python50k_eval.json",'r') as load_f:
    data1 = load_f.readlines()
for i in range(len(data1)):
    content = json.loads(data1[i])
    data.append(content)

# dfs flatten
data_flatten = []
for i  in range(len(data)):
    data_flatten .append(dfs_AST(data[i], 0))

# {type:value} form
eval_data= []
for datalist in data_flatten:
    inner_data = []
    for items in datalist:
        inner_data.append({items["type"]:items["value"]})
    eval_data.append(inner_data)


eval_data_rd = []
vocab_size = len(value_to_ix )
for i in range(len(eval_data)):
    length = len(eval_data[i])
    if length <= CONTEXT_WINDOW+1:
        continue
    rd = random.randint(CONTEXT_WINDOW + 1, len(eval_data[i]) - 1) #取rand int rd
    inner_data = [eval_data[i][:rd], [eval_data[i][rd]], -1, rd] #[context,global position, context position, self -position)
    for j in range(rd-1 , rd- CONTEXT_WINDOW,-1): #whether the remove node in the context.if the node in context,we remeber the position in context
        if j < 0:
            break  #some sequence length is less than CONTEXT_WINDOW
        if eval_data[i][rd] == eval_data[i][j]:
            inner_data = [eval_data[i][:rd], [eval_data[i][rd]], vocab_size+rd - j -1, rd]
            break
    eval_data_rd.append(inner_data)