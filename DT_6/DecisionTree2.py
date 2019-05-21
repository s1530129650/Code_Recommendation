#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: DecisionTree2.py
@time: 2019/5/13 18:57
position information
"""

from gene_queries_occur1 import gene_queries
from Tgen1 import TGEN
from collections import Counter
from probablistic_model1 import  traverse2model
import numpy as np
from GA_1 import GA_Algorithm
from sklearn.feature_extraction.text import HashingVectorizer
#from GA_ACCe import GA_Algorithm

from math import log
import random
import sys
import copy
import time
sys.setrecursionlimit(150000)  # set the maximum depth as 150
start_time = time.time()
import warnings

warnings.filterwarnings('ignore')

#print("len(quer_data)",len(quer_data))


#2 calulate entropy
def cal_entropy(data_cal):
    label_list =  []
    numEntries = len(data_cal)

    for i in range(numEntries):
        label_list.append(data_cal[i].label)
    labelCounts = Counter(label_list )
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#entropy = cal_entropy(quer_data)
#print("entorpy",entropy)

# 3 generate feature
def gene_feature(mv_instr,write_instr,feature_num):
    import itertools
    instr = []
    for wr in write_instr:
        instr.append(wr)
    for i in range(1, feature_num):
        iter1 = itertools.permutations(mv_instr, i)
        list_iter = list(iter1)
        for wr in write_instr:
            iter2 = [list(c) + wr for c in list_iter]
            instr= instr + list(iter2)
    return instr


# 4 split data
def split_set(universalSet, item):
    retDataSet = []
    i = 0
    #print("item",item)
    while i < (len(universalSet)):
        if universalSet[i].ctx[-1] == item:  #

            retDataSet.append(universalSet[i])
            universalSet.remove(universalSet[i])
            i = i - 1
        i = i + 1
    # print("split",len(retDataSet))
    #print("dataSet_ch", len(universalSet))
    return retDataSet


## 5 max gain ratio
def choose_feature_split(query_data,instr):
    fea_list = []
    base_entropy = cal_entropy(query_data)  # dataset entropy

    best_idx = GA_Algorithm(base_entropy ,query_data,instr)
    bestFeature = instr[best_idx]

    return best_idx,bestFeature

#best_feature = choose_feature_split(quer_data,instructions,write_instruction)

## 6 creat tree
length = [0]
def create_tree(all_data,instruc):

    label = []
    data_len = len(all_data)
    for data in all_data:
        data.resetNode()
        label.append(data.label)
    label_list = Counter(label)  # all_data label
    if data_len < MIN_SIZE:  # too smaller
        length[0] += 1
        return all_data
    if not instruc:
        length[0] += 1
        return all_data
    print("start choose_feature_split")
    #now1 = time.time()
    idx,p_best_feat = choose_feature_split(all_data, instruc)
    #now2 = time.time()
    #print("choose_feature_split",(now2- now1)/60)

    if not p_best_feat:
        length[0] += 1
        return all_data
   # print( p_best_feat,)
    bestFeatLabel = tuple(p_best_feat)
    myTree = {bestFeatLabel: {}}

    fea_list = []
    #len_list = []
    for data in all_data:
        data.geneProgram(p_best_feat)
        fea_list.append(data.ctx[-1])
        #len_list.append(data.length)
        #print(data.tree[-5:])
    uniqueCtx = Counter(fea_list)  # datasize feature
    uniqueCtx = uniqueCtx.most_common(10)


    #print(uniqueCtx)
    #print(len_list)
    for last_ctx in uniqueCtx:
        subLabels = copy.deepcopy(instruc)
        del subLabels[idx]
        myTree[bestFeatLabel][last_ctx[0]] = create_tree(split_set(all_data, last_ctx[0]), subLabels)
    if len(all_data) > 0:
        subLabels = copy.deepcopy(instruc)
        del subLabels[idx]
        myTree[bestFeatLabel]["DEFAULT:"] = create_tree(all_data, subLabels)


    return myTree

#myTree = create_tree(quer_data,instructions,write_instruction)

def get_result(claf,data):
    clf_DT = claf[0]
    clf_cache = claf[-1]
    data_ctx = data.cache
    length = len(data_ctx)
    Cache= {}
    for item in clf_cache:
        Cache[data_ctx[item[0] % length]] =item[1]
    for item in clf_DT:
        Cache.setdefault(item[0], 0)
        Cache[item[0]] = Cache[item[0]] + item[1]
    suggestion  = sorted(Cache.items(), key=lambda d: d[1], reverse=True)
    result = [key for key,value in suggestion[:5]]
    return result

# 7. code suggestion
def code_suggestion(inputTree, test_data,vectorizer):  # single test_data
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    test_data.geneProgram(firstStr)
    test_data.resetNode()
    suggestion = []
    flag =  False
    for key in secondDict.keys():  # secondDict.keys()＝[0, 1, 2]
        if test_data.ctx[-1] == key:  # secondDict[key]＝N
            flag = True
            if type(secondDict[key]).__name__ == 'dict':  # type(secondDict[key]).__name__＝str
                suggestion = code_suggestion(secondDict[key], test_data,vectorizer)
            else:
                claf = secondDict[key]
                #print(len(claf))
                if len(claf) <3 :
                    #print("unconditional probabilistic model")
                    suggestion = get_result(claf,test_data)
                else:
                    test_data.geneProgram(claf[0])
                    test_data.resetNode()
                    vector = vectorizer.transform([str(test_data.ctx[-1])])
                    data_vec = vector.toarray()
                    result = claf[1].predict_proba(data_vec )[0]
                    top_k = 5
                    #print("result",result)
                    sugge= result.argsort()[::-1][0:top_k ]
                    sug_prob = [result[i] for i in sugge]
                    sug_value = list(claf[2].inverse_transform(sugge))
                    dictionary = list(zip(sug_value , sug_prob ))
                    suggestion = get_result([dictionary,claf[-1]], test_data)

            break

    if not flag and "DEFAULT:" in secondDict.keys():
        key = "DEFAULT:"
        if type(secondDict[key]).__name__ == 'dict':  # type(secondDict[key]).__name__＝str
            suggestion = code_suggestion(secondDict[key], test_data, vectorizer)
        else:
            claf = secondDict[key]
            # print(len(claf))
            if len(claf) < 3:
                # print("unconditional probabilistic model")
                suggestion = get_result(claf, test_data)
            else:
                test_data.geneProgram(claf[0])
                test_data.resetNode()
                vector = vectorizer.transform([str(test_data.ctx[-1])])
                data_vec = vector.toarray()
                result = claf[1].predict_proba(data_vec)[0]
                top_k = 5
                # print("result",result)
                sugge = result.argsort()[::-1][0:top_k]
                sug_prob = [result[i] for i in sugge]
                sug_value = list(claf[2].inverse_transform(sugge))
                dictionary = list(zip(sug_value, sug_prob))
                suggestion = get_result([dictionary, claf[-1]], test_data)

    return suggestion
# tree = copy.deepcopy(myTree)
# tree1 = traverse2model(tree)


#9 . evaluation
def eval(model,eval_data,vectorizer):
    length = len(eval_data)
    #AP
    def AP(pre, ground_true):
        Ap = 0
        for i in range(len(pre)):
            if pre[i] == ground_true:
                Ap = Ap + 1 / (i + 1)
        return Ap

    A_P = []
    for data in eval_data:
        #start = time.time()
        result = code_suggestion(model,data,vectorizer)

        if not result:
            length = length - 1
            continue

        #result = sorted(res[0].items(), key=lambda d: d[1],reverse = True)
        A_P.append(AP( result, data.label))

    m_a_p = sum(A_P) /  length
    return m_a_p

#eval(tree1,testData)


MIN_SIZE = 500
print("MIN_SIZE",MIN_SIZE)

def main_tree():
    mv_instructions = ["mvpar", "mvLeftSibl", "mvRightSibl", "mvFirstChild", "mvLastChild", "mvPrevDFS", "mvNextDFS",
                       "mvPrevLeft", "mvNextLeft", "mvPrevNodeValue", "mvPrevNodeType","mvPrevNodeContext"]
    write_instruction = [["wrVal"],["wrType"]]


    #1. data load and shuffle
    quer_data = gene_queries()
    print("data len",len(quer_data))
    quer_data =  quer_data[:10000]
    #test data
    test_data = gene_queries(Test_flag=True)
    random.shuffle(test_data)  # shuffle
    test_data = test_data[:10000]

    now = time.time()
    print("1 data load",now- start_time)
    random.shuffle(quer_data) #shuffle

    split_pos = int(len(quer_data) * 0.7)
    training_data = quer_data[:split_pos]
    eval_data= quer_data[split_pos:]
    print("data set size", len(training_data), len(eval_data),len(test_data))

    # 2. make instructions set and shuffle
    feature_num = 5
    print("feature_num",feature_num)
    instructions = gene_feature(mv_instructions,write_instruction,feature_num)
    now = time.time()
    print("2 get feature",now- start_time)
    #random.shuffle(instructions)


    # 3. creat tree
    my_tree = create_tree(training_data,instructions)
    now = time.time()
    print("3 get myTree",now- start_time)
    #print(my_tree)
    # 4 . save tree
    #print(my_tree)
    import json
    import pickle

    #5. get probabilistic model
    vectorizer = HashingVectorizer(n_features=20, non_negative=True, )
    model_tree = traverse2model(my_tree ,instructions,vectorizer)
    now = time.time()
    print("5 model", now - start_time)
    #print("model",model_tree)
    with open('model.pickle', 'wb') as f1:
        pickle.dump( model_tree, f1, protocol=pickle.HIGHEST_PROTOCOL)


    #6 evaluate
    MAP = eval(model_tree, eval_data, vectorizer)
    print("eval MAP:", MAP)
    # 7 test
    MAP_test = eval(model_tree, test_data, vectorizer)
    print("test MAP:", MAP_test)

    now = time.time()
    m, s = divmod((now - start_time), 60)
    h, m = divmod(m, 60)
    print("number of classifier",length[0])
    print("6 evaluate time spend%02d:%02d:%02d: " % (h, m, s))


if __name__ == '__main__':
    main_tree()



