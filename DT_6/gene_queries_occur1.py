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


def gene_queries(Test_flag = False):

    use_gpu = False
    use_gpu = True

    if use_gpu:

        max_vocab_size = 50000
        min_snippet_len = 20
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
            if i > 2000:
                break


        return data


    # taining data
    if use_gpu:
        if Test_flag:
            training_path = r"../data/python/python50k_eval.json"
        else:
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
                pos = -1
                cache = []
                flag = True
                for j in range(i - 1, max(i - CONTEXT_WINDOW - 1,-1),-1):  #
                    if "value" in data[j]:
                        pos = pos + 1
                        cache.append(data[j]["value"])


                    if "value" in data[j] and data[j]["value"] == data[i]["value"] and flag:
                        flag = False
                        #DT+model
                        Tree = data[:i]
                        node = copy.deepcopy(data[i])
                        label = data[i].get("value")
                        node["value"] = "need_to_predict"
                        Tree.append(node)
                        # cache model
                        position = pos
                    if (not flag) and (j == max(i - CONTEXT_WINDOW, 0) or pos >= 15):
                        queries.append(TGEN(Tree, node, label, position, cache[:15]))
                        break
                        #print(Tree,label)
                if j == max(i - CONTEXT_WINDOW - 1,-1):  # there is no same node in context
                    continue


        return queries

    quer_data = make_queries(add_par_data)
    #print(len(quer_data))
    '''
        print(20*"*")
        print(quer_data[0])
        print(quer_data)
    '''
    return quer_data

