#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: ASt.py
@time: 2019/3/13 10:19
"""

import ast
root_node = ast.parse('a=1')

print(ast.dump(root_node))
Module(body=[Assign(targets=[Name(id='a', ctx=Store())], value=Num(n=1))])