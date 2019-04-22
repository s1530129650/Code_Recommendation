#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: testlosad.py
@time: 2019/4/19 16:41
"""
import json
import numpy as np


with open(r"..\data\python\f2.json", 'r') as load_f_again:
    data1 = load_f_again.readlines()

    data= json.loads(data1[0])

    print(data)

