#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: get_queries.py
@time: 2019/5/20 16:38
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

use_gpu = False
use_gpu = True

if use_gpu:
    device = torch.device("cuda")
    max_vocab_size = 20000
    CONTEXT_WINDOW = 50
    Max_Seq_Len = 2000
else:
    device = torch.device("cpu")
    max_vocab_size = 800
    CONTEXT_WINDOW = 50
    Max_Seq_Len = 1000

time_start = time.time()


##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
def data_loading(filepath):
    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        data.append(content)
        if i >5000:
            break

    return data


# taining data
if use_gpu:
    training_path = r"../data/python/python100k_train.json"
else:
    str = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
    training_path = str + r"\data\python\f20_.json"

training_data = data_loading(training_path)

# eval data

if use_gpu:
    eval_path = r"../data/python/python50k_eval.json"
else:
    str = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
    eval_path = str + r"\data\python\f10_.json"

eval_data = data_loading(eval_path)

now = time.time()
print("1.data loading", now - time_start)


## 2. build vocabulary
def build_vocab(data1, data2):
    type_to_ix = {"EOF": 0, "UNK": 1}
    word_to_ix = {}
    for i in range(len(data1)):
        for item in data1[i]:
            if item["type"] not in type_to_ix:
                type_to_ix[item["type"]] = len(type_to_ix)
            if "value" in item.keys():
                if item["value"] in word_to_ix:
                    word_to_ix[item["value"]] = word_to_ix[item["value"]] + 1
                else:
                    word_to_ix[item["value"]] = 1

    for i in range(len(data2)):
        for item in data2[i]:
            if item["type"] not in type_to_ix:
                type_to_ix[item["type"]] = len(type_to_ix)
            if "value" in item.keys():
                if item["value"] in word_to_ix:
                    word_to_ix[item["value"]] = word_to_ix[item["value"]] + 1
                else:
                    word_to_ix[item["value"]] = 1

    # 1k 10k  50k vocabulary
    L = sorted(word_to_ix.items(), key=lambda item: item[1], reverse=True)
    print("L len", len(L), L[max_vocab_size][1])
    value_to_ix = {"EOF": 0, "UNK": 1}
    for i in range(max_vocab_size):
        value_to_ix[L[i][0]] = len(value_to_ix)
    return type_to_ix, value_to_ix

type_vocab, value_vocab = build_vocab(training_data, eval_data)
print("vocab len: type:", len(type_vocab), "value:", len(value_vocab))

now = time.time()
print("2.build vocabulary", now - time_start)


# 3. make the queries
def Queries(Data):
    data_rd = []

    for data in Data:
        length = len(data)
        if length <= CONTEXT_WINDOW + 2  or length > 2000:
            continue
        for i in range(CONTEXT_WINDOW , length):  # make queries
            if "value" not in data[i]:  # fine terminal
                continue

            query = [data[:i], [data[i]], i]
            pos = -1
            flag = False
            for j in range(i - 1, max(i - CONTEXT_WINDOW - 1, -1), -1):  #
                if "value" in data[j]:
                    pos = pos + 1
                if "value" in data[j] and data[j]["value"] == data[i]["value"]:
                    query.append(pos)
                    flag = True
                    break
                    # print(Tree,label)

            # no same node in the context
            if not flag:
                continue

            # add parents node
            for j in range(pos - 1, pos - CONTEXT_WINDOW - 2, -1):

                if "children" in data[j].keys() and pos in data[j]["children"]:
                    query.append(pos - j - 1)
                    break
                if j == pos - CONTEXT_WINDOW:
                    query.append(CONTEXT_WINDOW - 1)
                    break
        # query = [context,   predict_node,   position, same_node_position, parent_node_position]
            data_rd.append(query)
    return data_rd


training_queries = Queries(training_data[:5000])
random.shuffle(training_queries) #shuffle
print("trn_eval len",len(training_queries))
trn_queries = training_queries[:200000]
print("trn_eval size",len(training_queries))


from sklearn.model_selection import train_test_split
train_queries , eval_queries = train_test_split(trn_queries, test_size = 0.3)

#training_queries.sort(key=lambda x: x[2], reverse=True)  # sort,baecause pack_padded_sequence
train_queries.sort(key=lambda x: x[2], reverse=True)  # sort,baecause pack_padded_sequence
eval_queries.sort(key=lambda x: x[2], reverse=True)  # sort,baecause pack_padded_sequence


#test_queries = Queries(eval_data)
test_queries =training_queries[200000:300000]
test_queries.sort(key=lambda x: x[2], reverse=True)  # sort,baecause pack_padded_sequence

# print(training_queries)

now = time.time()
print("3.make the queries", now - time_start)


# 4 text -> index

def prepare_sequence(seq, val_to_ix, type_to_ix):  # trans code to idex
    idxs_ty = []
    idxs_vl = []
    UNK = 1
    for node in seq:
        value_str = node.get('value', 'UNK')
        idxs_vl.append(val_to_ix.get(value_str, UNK))
        idxs_ty.append(type_to_ix[node.get('type', 'UNK')])
    # print("np.array([idxs_ty, idxs_vl])",np.array([idxs_ty, idxs_vl]))
    return  torch.tensor([idxs_vl, idxs_ty],dtype = torch.long)


def textToindex(prepare_sequence,queries,value_vocab,type_vocab):
    input_value = []
    input_type= []
    parent = []
    target = []
    for i in range(len(queries)):
        sequence = queries[i][0]
        [input_val, input_ty] = prepare_sequence(sequence, value_vocab, type_vocab)
        par = torch.tensor(queries[i][4], dtype=torch.long)
        targ = torch.tensor(queries[i][3], dtype=torch.long)

        input_value.append(input_val)
        input_type.append(input_ty)
        parent.append(par)
        target.append(targ)

    return input_value,input_type,parent,target

# training set -> train set


#input_value_train_all,input_type_train_all,parent_train_all,target_train_all = textToindex(prepare_sequence,training_queries,value_vocab,type_vocab)

input_value_train,input_type_train,parent_train,target_train = textToindex(prepare_sequence,train_queries,value_vocab,type_vocab)
input_value_eval,input_type_eval,parent_eval,target_eval = textToindex(prepare_sequence,eval_queries,value_vocab,type_vocab)
input_value_test,input_type_test,parent_test,target_test = textToindex(prepare_sequence,test_queries,value_vocab,type_vocab)

now = time.time()
print("4 text -> index", now - time_start)


# 5 padding and save
import torch.nn.utils.rnn as rnn_utils

import torch.utils.data as data

class MyData(data.Dataset):
    def __init__(self, data_seq, input_value, input_type, target, parent):
        self.input_value = input_value
        self.input_type = input_type
        self.target = target
        self.parent = parent
        self.length = len(self.target)
        self.data_length = [len(sq) for sq in data_seq]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.input_type[idx], self.input_value[idx], self.data_length[idx], self.target[idx], self.parent[
            idx]

import pickle
import gc
#all train
'''
x_train_all = rnn_utils.pad_sequence(input_value_train_all, batch_first=True)
y_train_all = rnn_utils.pad_sequence(input_type_train_all, batch_first=True)
dataAll_train_all = MyData(input_value_train_all, x_train_all, y_train_all, target_train_all, parent_train_all)
print(20*"#")
print("train_all_data_length",dataAll_train_all.length)

with open('../data/python/training3_50k.pickle', 'wb') as f3:
    pickle.dump(dataAll_train_all, f3, protocol=pickle.HIGHEST_PROTOCOL)

del dataAll_train_all
gc.collect()
'''
#train


x_train = rnn_utils.pad_sequence(input_value_train, batch_first=True)
y_train = rnn_utils.pad_sequence(input_type_train, batch_first=True)

dataAll_train = MyData(input_value_train, x_train, y_train, target_train, parent_train)

print("train_data_length",dataAll_train.length)

#print("make mydata time spend",end - start)
with open('../data/python/train3_50k.pickle', 'wb') as f1:
    pickle.dump(dataAll_train, f1, protocol=pickle.HIGHEST_PROTOCOL)


del dataAll_train
gc.collect()

# eval
x_eval = rnn_utils.pad_sequence(input_value_eval, batch_first=True)
y_eval = rnn_utils.pad_sequence(input_type_eval, batch_first=True)
dataAll_eval = MyData(input_value_eval, x_eval, y_eval, target_eval, parent_eval)
print("eval_data_length",dataAll_eval.length)

with open('../data/python/eval3_50k.pickle', 'wb') as f2:
    pickle.dump(dataAll_eval, f2, protocol=pickle.HIGHEST_PROTOCOL)

del dataAll_eval
gc.collect()

#test
x_test = rnn_utils.pad_sequence(input_value_test, batch_first=True)
y_test = rnn_utils.pad_sequence(input_type_test, batch_first=True)
dataAll_test = MyData(input_value_test, x_test, y_test, target_test, parent_test)
print("test_data_length",dataAll_test.length)
print(20*"#")

with open('../data/python/test3_50k.pickle', 'wb') as f3:
    pickle.dump(dataAll_test, f3, protocol=pickle.HIGHEST_PROTOCOL)

del dataAll_test
gc.collect()

now = time.time()
print("5. padding ", now - time_start)

# 6 save
np.savez('../data/python/vocabulary_50k.npz', value_vocab=value_vocab, type_vocab=type_vocab)
now = time.time()
print("6. save ", now - time_start)




