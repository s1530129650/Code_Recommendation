#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: visualize_json.py
@time: 2019/3/13 10:14
"""
import json
import pandas as pd
import matplotlib.pyplot as plt

str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"

with  open(str+"\data\python\fist_line_data",'r',encoding = 'utf8') as load_f:

    data = json.load(load_f)
    #print(type(data))
    print(data)
df = pd.DataFrame(data)
df.plot.box(title="picture",)
plt.grid(linestyle="--", alpha=0.3)
plt.savefig("test3.png",dpi=999)
plt.show()
