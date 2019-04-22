#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: json_in_out.py
@time: 2019/4/19 10:00
"""
import json



str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"

with open(str+"\data\python\python100k_train.json",'r') as load_f:
    data = load_f.readline()
    print(data) #10w


print("******************************************")
'''
with open(str+r"\data\python\first_line_data.json",'w') as store_f:
    json.dump(data,store_f)
    print("store data is done")
'''


with open(str+r"\data\python\first_line_data.json",'r') as load_f_again:
    data1 = load_f_again.readline()
    dic = json.loads(data1)
    print(dic)