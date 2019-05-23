#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: count.py
@time: 2019/4/19 16:11
"""
import json
import numpy as np
data = []
with open(r"../data/python/python50k_eval.json",'r') as load_f_again:
    #data1 = load_f_again.readlines()
    for line in load_f_again:
        dic = json.loads(line)
        data.append(dic)
    #print(data[0])

#data1 = data [0]
countList = []
for i in range(len(data)):
    countList.append(len(data[i]))
'''
max_value = np.max(countList)
min_value = np.min(countList)
avg_value = np.mean(countList)
var_value =  np.var(countList)
std_value = np.std(countList,ddof=1)
print("max:",max_value)
print("min:",min_value)
print("avg:",avg_value)
print("var:",avg_value)
print("std:",avg_value)

'''

a=np.sum(countList)
print("taining set")
print(a)




