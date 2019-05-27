#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: cache_model.py
@time: 2019/5/16 16:45
"""
import json
import copy
import time
import numpy as np

min_snippet_len = 20
CONTEXT_WINDOW = 50
use_gpu = False
use_gpu = True
start = time.time()
'''
##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
def data_loading(filepath):
    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        #data = data + [content[:i+min_snippet_len] for i in range(len(content)-min_snippet_len-1)]
        data.append(content)

    return data


# taining data
if use_gpu:
    training_path = r"../data/python/python100k_train.json"

else:
    str1 = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
    training_path = str1 + r"\data\python\f50_.json"

training_data = data_loading(training_path)
#2 .make queries
def make_queries(dataList):
    pos_label = []
    node_type = []
    for data in dataList:  # one AST
        length = len(data)
        if length < min_snippet_len or length > 2000:  # quit too small sample
            continue

        for i in range(min_snippet_len, length):  # make queries
            if "value" not in data[i]:  # fine terminal
                continue

            # find same terminal  in the context

            # print("i",i)
            pos = 0
            for j in range(i - 1, max(i - CONTEXT_WINDOW - 1, -1), -1):  #
                if "value" in data[j]:
                    pos = pos + 1
                if "value" in data[j] and data[j]["value"] == data[i]["value"]:
                    pos_label.append(pos)
                    node_type.append({"non_terminal": data[i]["type"]})

                    break
                    # print(Tree,label)
            if j == max(i - CONTEXT_WINDOW - 1, -1):  # there is no same node in context
                continue

    return pos_label, node_type
'''


with np.load(r"count.npz", allow_pickle=True) as arr:
    position_label= arr['position_occur']


#pos_label, node_type = make_queries(training_data)

#3 model
position_label = position_label[:8000000]

split_pos = int(len(position_label) * 0.7)
training_pos= position_label[:split_pos]
eval_pos = position_label[split_pos:]
from collections import Counter
fre = Counter(training_pos).most_common(5)
recom = []


for item in fre:
    recom.append(item[0])

print("len",len(training_pos),len(eval_pos))
# evaluate
def AP(pre, ground_true):
    Ap = 0
    for i in range(len(pre)):
        if pre[i] == ground_true:
            Ap = Ap + 1 / (i + 1)
    return Ap
A_P = []
for k in range(len(eval_pos)):

    A_P.append(AP(  recom, eval_pos[k]))
m_a_p = sum(A_P) /  len(eval_pos)
print("MAP",m_a_p)
end = time.time()
print("time spend:",end- start)