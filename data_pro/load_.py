#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: load_.py
@time: 2019/4/19 10:08
"""

# -*- coding: utf-8 -*-
import json
import codecs
"""
data = []
with codecs.open(r"..\data\python\python100k_train.json", "r") as f:
    data1 = f.readlines()
    for line in data1[:2]:
        dic = json.loads(line)
        data.append(dic)
        print (json.dumps(dic, indent=4, ensure_ascii=False))

with codecs.open(r"..\data\python\f2.json",'w', 'utf-8') as outf:
    json.dump(data, outf, ensure_ascii=False)
    outf.write('\n')


data = []
with open(r"..\data\python\f2.json",'r') as load_f_again:
    #data1 = load_f_again.readlines()
    for line in load_f_again:
        dic = json.loads(line)
        data.append(dic)
    print(data)



with open(r"..\data\python\f2.json",'r') as load_f_again:
    data1 = load_f_again.readlines()
    data = json.loads(data1[0])

    print(data)"""

data = []
with codecs.open(r"..\data\python\python100k_train.json", "r") as f:
    data1 = f.readlines()
    for line in data1[:20]:
        dic = json.loads(line)
        data.append(dic)
        print (json.dumps(dic, indent=4, ensure_ascii=False))

with codecs.open(r"..\data\python\f20_.json",'w', 'utf-8') as outf:
    for items in data:
        json.dump(items, outf, ensure_ascii=False)
        outf.write('\n')


