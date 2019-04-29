#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: evaluate.py
@time: 2019/4/28 21:22
"""

def AP(data):
    Ap = 0
    for i in range(len(data)):
        if data[i] == 1:
            Ap = Ap + 1/(i+1)
    return Ap
def MAP(data):
    A_P = 0
    for i in range(len(data)):
        A_P  +=  AP(pre[i])
    mAp = A_P / len(data)
    return mAp

pre = [[0,1,0,0,0],[1,0,0,0,0]]
MAP = MAP(pre)
print(MAP)
