#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: N_GRAM.py
@time: 2019/5/20 17:36
"""

import keyword
import json
from nltk import ngrams
import copy
from collections import Counter
import heapq
import random
import time

start = time.time()

## 1 . data loading
def data_loading(filepath):
    data = []
    with open(filepath, 'r') as load_f:
        data1 = load_f.readlines()
    for i in range(len(data1)):
        content = json.loads(data1[i])
        # data = data + [content[:i+min_snippet_len] for i in range(len(content)-min_snippet_len-1)]
        data.append(content)
        if i > 400:
            break

    return data
training_path = r"../data/python/python100k_train.json"
training_data = data_loading(training_path)
now = time.time()
print("1.data_loading ", (now - start)/60)

# 2. tree -> sequence
def UnParse(Data):
    seq = []
    KWList = keyword.kwlist
    for item in Data:
        if item["type"].lower() in KWList:
            seq.append(item["type"].lower())
        if 'children' not in item:
            seq.append(item.get("value","EMPTY"))
    return seq

sequence = []
for data in training_data:
    sequence.append(UnParse(data))

#print(sequence[0][:10])
now = time.time()
print("2. tree -> sequence ", (now - start)/60)


# 3 n-gram model
def NGram(dataSet,n):
    fea = []
    for data in dataSet:
        n_gram = ngrams(data, n)
        for item in  n_gram:
            fea.append(item)
    return fea



# vocabulary
feature1 = NGram(sequence,1)
gram_1  = list(set(feature1))
vocab = [ item[0] for item in gram_1]
count = Counter(feature1).most_common(5)
#print(vocab)

suggest = []
for k,v in count:
    #print(k,v)
    suggest.append(vocab.index(list(k)[0]))
#print("suggest",suggest)
#2-gram
feature2 = NGram(sequence,9)
gram_2  = dict(Counter(feature2))

#3-gram
feature3 = NGram(sequence,10)
gram_3  = dict(Counter(feature3))
print("traing len",len(gram_3))
model = {}
for item in gram_2:
    value_list = []
    list_item = list(item)
    for word in vocab:

        ls_item = copy.deepcopy(list_item)
        ls_item.append(word)
        value = gram_3.get(tuple(ls_item),random.uniform(0.1,0.9))

        value_list.append(value)
    #print("value_list",value_list)
    max_num_index_list = map(value_list.index, heapq.nlargest(5, value_list))
    #print( "max_num_index_lis",list(max_num_index_list))
    model[item] = list(max_num_index_list)

#print(model)
now = time.time()
print("3. build n-gram model ", (now - start)/60)

# 4 eval
test_path = r"../data/python/python50k_eval.json"
test_data = data_loading(test_path)
seq_test = []
for data in test_data:
    seq_test.append(UnParse(data))
# 4.1 make Queries
def make_queries(dataSet,n):
    query = []
    for data in dataSet:
        n_gram = ngrams(data, n)
        for item in  n_gram:
            ls_it = list(item )
            query.append([tuple(ls_it[:-1]),ls_it[-1]])
    return query
Queries = make_queries(seq_test,10)
print("Queries len", len(Queries ))
now = time.time()
print("4.1 make Queries ", (now - start)/60)
# 4.2 metric
def AP(pre, ground_true,vocab):
    Ap = 0
    for i in range(len(pre)):
        if vocab[pre[i]] == ground_true:
            Ap = Ap + 1 / (i + 1)
            break

    return Ap

#4.2 suggestion
A_P = []
length = len(Queries)

for Query in Queries:
    res = model.get(Query[0],"UNKK")
    if res == "UNKK":
        res = suggest
    A_P.append(AP(res,Query[1],vocab))
m_a_p = sum(A_P) /  length
print("MAP",m_a_p )
now = time.time()
print("4  done ", (now - start)/60)



