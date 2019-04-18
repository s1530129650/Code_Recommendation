#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: load_data.py
@time: 2019/3/13 9:23
"""
import json

str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"

with open(str+"\data\python\python100k_train.json",'r') as load_f:
    data = load_f.readlines()


print(len(data))
print((data[0]))
content = json.loads(data[0])
#print((data[0][0:-1]))
print(type(content))
print(content)
print(content[0])

'''

for i in range(len(data)):
    content.json.loads(data[i][0])

print(len(content))

'''