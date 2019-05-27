#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: COUNT.py
@time: 2019/5/16 14:40
"""
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
import time
import numpy as np

min_snippet_len = 10
CONTEXT_WINDOW = 1000
use_gpu = False
use_gpu = True


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

def make_queries(dataList):

    cache_ctx = []

    for data in dataList:  # one AST
        length = len(data)
        if length < min_snippet_len or length > 2000:  # quit too small sample
            continue

        for i in range(min_snippet_len, length):  # make queries
            if "value" not in data[i]:  # fine terminal
                continue

            # find same terminal  in the context

            # print("i",i)
            posi = 0
            for j in range(i - 1, max(i - CONTEXT_WINDOW - 1, -1), -1):  #
                if "value" in data[j]:
                    posi = posi + 1
                   # print(pos,data[j]["value"],data[i]["value"])
                if "value" in data[j] and data[j]["value"] == data[i]["value"]:
                    cache_ctx.append(posi)

                    break
                    # print(Tree,label)
            if j == max(i - CONTEXT_WINDOW - 1, -1):  # there is no same node in context
                continue

    return cache_ctx
countList = make_queries(training_data)
'''


with np.load(r"count.npz", allow_pickle=True) as arr:
    countList = arr['position_occur']

'''

max_value = np.max(countList)
min_value = np.min(countList)
avg_value = np.mean(countList)
var_value =  np.var(countList)
std_value = np.std(countList,ddof=1)
length = len(countList)
print("count",length)

length = len(countList)
print("count",length)
print("500",1 - sum(i > 500 for i in countList)/length)
print("100",1 -sum(i > 100 for i in countList)/length)
print("50",1 - sum(i > 50 for i in countList)/length)
print("20",1 - sum(i > 20 for i in countList)/length)
print("15",1 - sum(i > 15 for i in countList)/length)
print("max:",max_value)
print("min:",min_value)
print("avg:",avg_value)
print("var:",avg_value)
print("std:",avg_value)

'''
print("大于40",sum(i > 40 for i in countList))
print("大于30",sum(i > 30 for i in countList))
print("大于25",sum(i > 25 for i in countList))
print("大于20",sum(i > 20 for i in countList))
print("大于15",sum(i > 15 for i in countList))
print("大于10",sum(i > 10 for i in countList))
print("max:",max_value)
print("min:",min_value)
print("avg:",avg_value)
print("var:",avg_value)
print("std:",avg_value)
'''
np.savez('count.npz',position_occur = countList)


