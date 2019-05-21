#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: probablistic_model1.py
@time: 2019/5/14 9:46
"""


from gene_queries_occur1 import gene_queries
from Tgen1 import TGEN

from collections import Counter
from math import log
import random
import sys
import copy
import time

from collections import Counter

from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from GA_leaf import GA_Algorithm


def Mult_NB(train_vec, labels):
    clf_NB = MultinomialNB().fit(train_vec, labels)
    return  clf_NB

def SVM_clf(train_vec, labels):
    clf_svm = svm.SVC(gamma='scale', probability=True)
    clf_svm.fit(train_vec, labels)

    return clf_svm

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

## 5 max gain ratio
def choose_feature_split(query_data,instr):
    fea_list = []
    base_entropy = cal_entropy(query_data)  # dataset entropy

    best_idx = GA_Algorithm(base_entropy ,query_data,instr)
    bestFeature = instr[best_idx]

    return best_idx,bestFeature

def Uncond_probab_model(labelCounts):

    leng = len(labelCounts)
    pr_dict = {}
    for items in labelCounts.most_common(5):
        pr_dict[items[0]]  = items[1]/leng
    result = sorted( pr_dict.items(), key=lambda d: d[1], reverse=True)
    pr_dict1 = []
    for k,v in result:
        pr_dict1.append(v)
    return  pr_dict1

def traverse2model(my_tree, instruc,vectorizer):
    firstStr = list(my_tree.keys())[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # type(secondDict[key]).__name__＝str
            classLabel = traverse2model(secondDict[key], instruc,vectorizer)
        else:
            leaf_data = secondDict[key]
            idx, p_best_feat = choose_feature_split(leaf_data, instruc)

            # 1 .data load lead_data
            train_data = []
            train_label = []
            # 2. get data and lable
            for data in leaf_data:
                data.geneProgram(p_best_feat)
                train_data.append(str(data.ctx[-1]))
                train_label.append(data.label)
            # 3 data: text -> num ->vect
            #print(len(leaf_data),train_data)
            labelCounts = Counter(train_label)
            if len(labelCounts) <= 5:
                secondDict[key] = [Uncond_probab_model(labelCounts)]
                continue
            vector = vectorizer.transform(train_data)
            train_vec = vector.toarray()
            #print("train_vec ",train_vec.shape )
            # 4 lable: text -> num
            le = preprocessing.LabelEncoder()
            le.fit(train_label)
            labels = le.transform(train_label)
            # 5 choose one clf

            secondDict[key] = [p_best_feat,Mult_NB(train_vec, labels),le]
            #secondDict[key] = Uncond_probab_model(key_data)
    return my_tree

