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
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

'''
def Mult_NB(leaf_data):
    # 1 .data load lead_data
    train_data = []
    train_label = []
    # 2. get data and lable
    for data in leaf_data:
        train_data.append(str(data.ctx[0]))
        train_label.append(data.label)

    # 3 data: text -> num ->vect
    # from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(n_features=5)
    vector = vectorizer.transform(train_data)
    train_vec = vector.toarray()

    # 4 lable: text -> num
    # from collections import Counter
    labelCounts = Counter(train_label)
    # from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    labels = le.transform(train_label)
    # print(labels)
    # list(le.inverse_transform([ 9, 34, 29, 30]))
    # 5 MultinomialNB

    clf = MultinomialNB().fit(train_vec, labels)

    return clf


def SVM_clf(train_vec, labels):
    clf_svm = svm.SVC(gamma='scale', probability=True)
    clf_svm.fit(train_vec, labels)

    return clf_svm
'''

def Uncond_probab_model(block_data):
    label = []
    for data_co in block_data:
        label.append(data_co.label)
    labelCounts = Counter(label)
    leng = len(block_data)
    pr_dict = {}
    for items in labelCounts.most_common(5):
        pr_dict[items[0]] = items[1] / leng
    return [pr_dict]


def traverse2model(my_tree):
    firstStr = list(my_tree.keys())[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # type(secondDict[key]).__name__＝str
            classLabel = traverse2model(secondDict[key])
        else:
            key_data = secondDict[key]

            secondDict[key] = Uncond_probab_model(key_data)
    return my_tree


'''
def main_pm():
    myTree = main_tree()
    #print("myTree",myTree)
    model = traverse2model(myTree)
    print("model",model)

main_pm()
'''
