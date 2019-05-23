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

from GA_1 import GA_Algorithm
#from GA_ACCe import GA_Algorithm

from math import log
import random
import sys
import copy
import time
sys.setrecursionlimit(150000)  # set the maximum depth as 150
start_time = time.time()

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
    for data_sp in universalSet:
        if data_sp.ctx[-1] == item:  #
            retDataSet.append(data_sp)
    # print("split",len(retDataSet))
    return retDataSet


## 5 max gain ratio
def choose_feature_split(query_data,instr):
    fea_list = []
    base_entropy = cal_entropy(query_data)  # dataset entropy

    best_idx = GA_Algorithm(base_entropy ,query_data,instr)
    bestFeature = instr[best_idx]
    '''
    for mvlist in instr:
        for wr in write_instr:
            Pfeature = list(mvlist) + wr
            #print("feature",Pfeature)
            dataSet_ch =  copy.deepcopy(query_data)
            for data_raw  in dataSet_ch:
                data_raw .geneProgram(Pfeature)
                fea_list.append(data_raw .ctx[-1])
                uniqueCtx_ch = Counter(fea_list )
            newEntropy = 0.0
            splitInfo = 0.0


            for last_ctx in uniqueCtx_ch:
                subDataSet = split_set( dataSet_ch, last_ctx)  #
                prob = len(subDataSet) / float(len( dataSet_ch))
                newEntropy += prob * cal_entropy(subDataSet)
                if prob == 0 :
                    splitInfo = 0
                else:
                    splitInfo += -prob * log(prob, 2)

            infoGain = baseEntropy - newEntropy  #  feature  infoGain
            if (splitInfo == 0):  # fix the overflow bug
                continue
            infoGainRatio = infoGain / splitInfo  # feature infoGainRatio
            if (infoGainRatio > bestInfoGainRatio):  #  gain ratio

                bestInfoGainRatio = infoGainRatio
                bestFeature = Pfeature  # gain ratio feature
                bestMvlist = mvlist
    '''
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
    now1 = time.time()
    idx,p_best_feat = choose_feature_split(all_data, instruc)
    now2 = time.time()
    print("choose_feature_split",(now2- now1)/60)

    if not p_best_feat:
        length[0] += 1
        return all_data
    #print( p_best_feat,Mvlist)
    bestFeatLabel = tuple(p_best_feat)
    myTree = {bestFeatLabel: {}}

    fea_list = []
    for data in all_data:
        data.geneProgram(p_best_feat)
        fea_list.append(data.ctx[-1])
        uniqueCtx = Counter(fea_list)  # datasize feature
    for last_ctx in uniqueCtx:
        subLabels = copy.deepcopy(instruc)
        del subLabels[idx]
        #subLabels.remove(Mvlist)
        #print(instruc)
        #print(20*"%")
        myTree[bestFeatLabel][last_ctx] = create_tree(split_set(all_data, last_ctx), subLabels)

    return myTree

#myTree = create_tree(quer_data,instructions,write_instruction)


# 7. code suggestion
def code_suggestion(inputTree, test_data):  # single test_data
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    test_data.geneProgram(firstStr)
    test_data.resetNode()
    suggestion = []
    for key in secondDict.keys():  # secondDict.keys()＝[0, 1, 2]
        if test_data.ctx[-1] == key:  # secondDict[key]＝N
            if type(secondDict[key]).__name__ == 'dict':  # type(secondDict[key]).__name__＝str
                suggestion = code_suggestion(secondDict[key], test_data)
            else:
                suggestion = secondDict[key]

    return suggestion
# tree = copy.deepcopy(myTree)
# tree1 = traverse2model(tree)

#7 . evaluation
def eval(model,eval_data):
    length = len(eval_data)
    #AP
    def AP(pre, ground_true):
        Ap = 0
        for i in range(len(pre)):
            if pre[i][0] == ground_true:
                Ap = Ap + 1 / (i + 1)
        return Ap

    A_P = []
    for data in eval_data:
        #start = time.time()
        res = code_suggestion(model,data)
        if not res:
            length = length - 1
            continue

        result = sorted(res[0].items(), key=lambda d: d[1],reverse = True)
        A_P.append(AP( result, data.label))
    m_a_p = sum(A_P) /  length
    return m_a_p

#eval(tree1,testData)


MIN_SIZE = 50

def main_tree():
    mv_instructions = ["mvpar", "mvLeftSibl", "mvRightSibl", "mvFirstChild", "mvLastChild", "mvPrevDFS", "mvNextDFS",
                       "mvPrevLeft", "mvNextLeft", "mvPrevNodeValue", "mvPrevNodeType","mvPrevNodeContext"]
    write_instruction = [["wrVal"],["wrType"]]


    #1. data load and shuffle
    value_vocab,type_vocab,quer_data = gene_queries()

    now = time.time()
    print("1 data load",now- start_time)
    random.shuffle(quer_data) #shuffle

    split_pos = int(len(quer_data) * 0.7)
    training_data = quer_data[:split_pos]
    test_data = quer_data[split_pos:]


    # 2. make instructions set and shuffle
    feature_num = 3
    print("feature_num",feature_num)
    instructions = gene_feature(mv_instructions,write_instruction,feature_num)
    now = time.time()
    print("2 get feature",now- start_time)
    #random.shuffle(instructions)
    print("data set size",len(training_data),len(test_data))

    # 3. creat tree
    my_tree = create_tree(training_data,instructions)
    now = time.time()
    print("3 get myTree",now- start_time)

    # 4 . save tree
    #print(my_tree)
    import json
    import pickle
    with open('myTree2.pickle', 'wb') as f1:
        pickle.dump(my_tree, f1, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('myTree.json', 'wb') as outf:
        #json.dump(my_tree, outf, ensure_ascii=False)
    now = time.time()
    print("4 save tree", now - start_time)

    #5. get probabilistic model
    model_tree = traverse2model(my_tree )
    now = time.time()
    print("5 model", now - start_time)

    with open('model.pickle', 'wb') as f1:
        pickle.dump( model_tree, f1, protocol=pickle.HIGHEST_PROTOCOL)


    #6 evaluate
    MAP = eval(model_tree,test_data)
    print("MAP:",MAP)
    now = time.time()
    m, s = divmod((now - start_time), 60)
    h, m = divmod(m, 60)
    print("number of classifier",length[0])
    print("6 evaluate time spend%02d:%02d:%02d: " % (h, m, s))


if __name__ == '__main__':
    main_tree()



