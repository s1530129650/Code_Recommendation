#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: dfs.py
@time: 2019/3/13 13:00
"""

'''
this function is used to ﬂatten each AST as a sequence of nodes in the in-order depth-ﬁrst traversal

'''
def dfs_AST(ast_list1, start):
    ast_list =ast_list1
    flatten_ast_list = []
    visited = [0]
    stack = [[start, 0]]

    while stack:
        note_name = "children"
        (v, next_child_idx) = stack[-1]
        if (note_name not in ast_list[v].keys()) or (next_child_idx >= len(ast_list[v]["children"])):

            stack.pop()
            continue
        next_child = ast_list[v]["children"][next_child_idx]
        stack[-1][1] += 1
        if next_child in visited:
            continue
        visited.append(next_child)
        stack.append([next_child, 0])
    #print(visited)
    for i in visited:
       # print(i,"children"  in ast_list[i].keys())

        if "children" in ast_list[i].keys():
            #print(ast_list[i])
            del(ast_list[i]['children'])
        if "value" not in ast_list[i].keys():
                #ast_list[i].pop(note_name)
            ast_list[i]["value"] = "empty"

        flatten_ast_list.append(ast_list[i])
    #print(flatten_ast_list)
    return flatten_ast_list


'''

data = [ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num"} ]#,"value":"1"

data_flatten = dfs_AST(data, 0)
print(data_flatten)



import json
data = []
str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
with open(str+"\data\python\python100k_train.json",'r') as load_f:
    data1 = load_f.readline()

content = json.loads(data1)
data.append(content)
print(data)
data_flatten = dfs_AST(data[0], 0)
print(data_flatten)
print(data_flatten[3])'''