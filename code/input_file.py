#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: input_file.py
@time: 2019/3/13 9:23
"""
import json



str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"

with open(str+"\data\python\python100k_train.json",'r') as load_f:
#with  open(str+r"\data\python\test.txt",'r',encoding = 'utf8') as load_f:
    data = load_f.readline()
    print(data) #10w
    #print(data)
#f = open(str+"\data\python\python100k_train.json",'r')
#f.close()
'''
content = json.loads(data[0])
print(type(content))
print(content)
'''
#data_0 = dict(content)
#print(data_0)



with open(str+r"\data\python\first_line_data.json",'w') as store_f:
    json.dump(data,store_f)
    print("store data is done")



