#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: gene_queries_occur1.py
@time: 2019/5/8 12:41
"""

import json
import copy
from Tgen1 import TGEN
import time


def gene_queries():

    use_gpu = False
    #use_gpu = True

    if use_gpu:

        max_vocab_size = 50000
        min_snippet_len = 3
        type_fre = 130
        CONTEXT_WINDOW = 50
    else:

        max_vocab_size = 418
        min_snippet_len = 3
        type_fre = 1
        CONTEXT_WINDOW = 30



    ##1. data loading {"type":xxx, "children":XXX} or {"type":xxx, "value":XXX}
    def data_loading(filepath):
        data = []
        with open(filepath, 'r') as load_f:
            data1 = load_f.readlines()
        for i in range(len(data1)):
            content = json.loads(data1[i])
            #data = data + [content[:i+min_snippet_len] for i in range(len(content)-min_snippet_len-1)]
            data.append(content)
            if i > 20:
                break


        return data


    # taining data
    if use_gpu:
        training_path = r"../data/python/python100k_train.json"

    else:
        str1 = r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
        training_path = str1 + r"\data\python\f50_.json"

    training_data = data_loading(training_path)

    '''
    print(20*"*")
    print(training_data[:5])
    print(20*"*")
    '''

    ##2. add parent node
    def add_parent(dataList):#[ [{},{}], [] ]
        data_par = copy.deepcopy(dataList)
        for data in data_par: #[{},{}]

            for i in range(len(data)): #{ "children":[]}

                if "children" in data[i]:
                    child_list = data[i]["children"]       # []
                    for index in child_list:

                        data[index]["parent"] = i

        return data_par

    add_par_data = add_parent(training_data)
    '''
    print(20*"*")
    print(training_data[:5])
    print(add_par_data[0][:5])
    print(20*"*")
    '''

    ## 3. build vocabulary
    def build_vocab(data):
        type_to_ix = {}
        word_to_ix = {}
        for i in range(len(data)):
            for item in data[i]:
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
        value_to_ix = {"EOF": 0,"UNK":1,"EMPTY":3}
        for i in range(len(L)):
            value_to_ix[L[i][0]] = len(value_to_ix)

        L1 = sorted(type_to_ix.items(), key=lambda item: item[1], reverse=True)
        type_to_ix = {"EOF": 0, "UNK": 1}
        type_comm = {"EOF": 0, "UNK": 1}
        type_comm = []
        for i in range(len(L1)):
            type_to_ix[L1[i][0]] = len(type_to_ix)

        return type_to_ix,value_to_ix

    type_vocab,value_vocab = build_vocab(training_data)

    ## 4 make queries
    def make_queries(dataList):
        queries = []
        for data in dataList: # one AST
            length = len(data)
            if length  < min_snippet_len or length > 2000: #  quit too small sample
                continue

            for i in range(min_snippet_len,length ): # make queries
                if "value" not in data[i]: #fine terminal
                    continue
                # find same terminal  in the context
                for j in range(i - 1, max(i - CONTEXT_WINDOW - 1,-1),-1):  #
                    if "value" in data[j] and data[j]["value"] == data[i]["value"]:
                        Tree = data[:i]
                        node = data[i]
                        label = data[i].get("value")
                        node["value"] = "need_to_predict"
                        Tree.append(node)
                        queries.append(TGEN(Tree, node, label))
                        break
                        #print(Tree,label)
                if j == max(i - CONTEXT_WINDOW - 1,-1):  # there is no same node in context
                    continue


        return queries

    quer_data = make_queries(training_data)
    #print(len(quer_data))
    '''
        print(20*"*")
        print(quer_data[0])
        print(quer_data)
    '''
    return value_vocab,type_vocab,quer_data

