#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: Tgen1.py
@time: 2019/5/7 14:25
"""
class TGEN:

    def __init__(self,tree,node,context = []):
        self.tree = tree
        self.node = node
        self.ctx = context
        self.length = len(tree)

    def writeValue(self,):
        val = self.node.get("value", "EMPTY")
        self.ctx.append(val)

    def writeType(self,):

        typ = self.node.get('type', 'UNK')
        self.ctx.append(typ)

    def writePos(self,):

        par = self.node.get("parent")              #parent node position
        nodeIndex =  self.tree.index(node)         #node position
        children = self.tree[par].get("children")  #childen list of parent of the node
        pos = children.index(nodeIndex)       #get the node index in it's parent's childrens nodes
        self.ctx.append(pos)

    def mvPar(self,): #move to the parent of node
        par = self.node.get("parent")# parent node position
        self.node = self.tree[par]

    def mvLeftSibl(self): #move to left sibling (its a circle)
        par = self.node.get("parent")  # parent node position)
        nodeIndex = self.tree.index(self.node)  # node position
        children = self.tree[par].get("children")  # childen list of parent of the node
        pos = children.index(nodeIndex)  # get the node index in it's parent's childrens nodes
        if pos > 0:
            self.node = self.tree[children[pos -1]]

    def mvRightSibl(self): #move to right sibling (its a circle)
        par = self.node.get("parent")  # parent node position
        nodeIndex = self.tree.index(self.node)  # node position
        children = self.tree[par].get("children")  # childen list of parent of the node
        pos = children.index(nodeIndex)  # get the node index in it's parent's childrens nodes
        if  pos + 1 < self.length  :
            self.node = self.tree[children[pos + 1] ]

    def mvFirstChild(self):
        nodeIndex = self.tree.index(self.node)  # node position
        children = self.node.get("children",[nodeIndex])
        self.node = self.tree[children[0]]

    def mvLastChild(self):
        nodeIndex = self.tree.index(self.node)  # node position
        children = self.node.get("children",[nodeIndex])
        self.node = self.tree[children[-1]]


    def mvPrevDFS(self):
        nodeIndex = self.tree.index(self.node)  # node position
        self.node = self.tree[nodeIndex - 1]

    def mvNextDFS(self):
        nodeIndex = self.tree.index(self.node)  # node position
        if nodeIndex + 1 < self.length :
            self.node = self.tree[nodeIndex + 1]

    def mvPrevLeft(self):
        nodeIndex = self.tree.index(self.node)
        for i in range(nodeIndex-1,-1,-1):
            if "children" not in self.tree[i]:
                self.node = self.tree[i]
        if i == 0 :
            self.node = self.tree[nodeIndex]

    def mvNextLeft(self):
        nodeIndex = self.tree.index(self.node)
        if nodeIndex  >= self.length - 1:
            self.node = self.tree[nodeIndex]
            return
        for i in range(nodeIndex , self.length, 1):
            if "children" not in self.tree[i]:
                self.node = self.tree[i]
        if i == self.length - 1:
            self.node = self.tree[nodeIndex]

    def mvPrevNodeValue(self):
        nodeIndex = self.tree.index(self.node)
        for i in range(nodeIndex - 1, -1, -1):
            if self.tree[i].get("value",0) == self.node.get("value",1):
                self.node = self.tree[i]
        if i == 0:
            self.node = self.tree[nodeIndex]

    def mvPrevNodeType(self):
        nodeIndex = self.tree.index(self.node)
        for i in range(nodeIndex - 1, -1, -1):
            if self.tree[i].get("type",0) == self.node.get("type",1):
                self.node = self.tree[i]
        if i == 0:
            self.node = self.tree[nodeIndex]
    def mvPrevNodeContext(self):
        nodeIndex = self.tree.index(self.node)
        nodePar = self.node.get("parent")
        for i in range(nodeIndex - 1, -1, -1):
            if self.tree[i].get("value",0) == self.node.get("value",1) and self.tree[i].get("type",0) == self.node.get("type",1) :
                newNodePar =  self.tree[i].get("parent",0)
                if self.tree[nodePar].get("value",0) == self.tree[newNodePar].get("value",1) and \
                        self.tree[nodePar].get("type",0) == self.tree[newNodePar].get("type",1):
                    self.node = self.tree[i]
        if i == 0:
            self.node = self.tree[nodeIndex]

    def geneProgram(self,proList):
        numbers = {
            "wrVal": self.writeValue,
            "wrTy": self.writeType,
            "wrPos": self.writePos,
            "mvpar": self.mvPar,
            "mvLeftSibl": self.mvLeftSibl,
            "mvRightSibl": self.mvRightSibl,
            "mvFirstChild": self.mvFirstChild,
            "mvLastChild": self.mvLastChild,
            "mvPrevDFS": self.mvPrevDFS,
            "mvNextDFS": self.mvNextDFS,
            "mvPrevLeft": self.mvPrevLeft,
            "mvNextLeft": self.mvNextLeft,
            "mvPrevNodeValue": self.mvPrevNodeValue,
            "mvPrevNodeType": self.mvPrevNodeType,
            "mvPrevNodeContext": self.mvPrevNodeContext,
        }
        for item in proList:
            method = numbers.get(item)
            if method:
                method()


Tree = [{'type': 'Module', 'children': [1, 3, 5, 7, 9, 11]}, {'type': 'Expr', 'children': [2], 'parent': 0}, \
        {'type': 'Str', 'value': ' Provides ``mapping`` of url paths to request handlers.\n', 'parent': 1}, \
        {'type': 'ImportFrom', 'children': [4], 'value': 'bootstrap', 'parent': 0}, {'type': 'alias', 'value': 'Bootstrap', 'parent': 3}]
node = Tree[-1]

print(Tree)
print(node)

Tgen = TGEN(Tree,node)
print(Tgen.ctx)
print(20*"#")
proList = ["wrVal","wrTy","wrPos"]
Tgen.geneProgram(proList)
print(Tgen.ctx)
'''


Tgen.mvPar()
print(Tgen.node)
print(20*"#")
Tgen.mvLeftSibl()
print(Tgen.node)
print(20*"#")
Tgen.mvRightSibl()
print(Tgen.node)

print(20*"#")
Tgen.mvFirstChild()
print(Tgen.node)

print(20*"#")
Tgen.mvLastChild()
print(Tgen.node)

print(20*"#")
Tgen.mvPrevDFS()
print(Tgen.node)

print(20*"#")
Tgen.mvNextDFS()
print(Tgen.node)

print(20*"#")
Tgen.mvPrevLeft()
print(Tgen.node)

print(20*"#")
Tgen.mvNextLeft()
print(Tgen.node)

print(20*"#")
Tgen.mvPrevNodeValue()
print(Tgen.node)

print(20*"#")
Tgen.mvPrevNodeType()
print(Tgen.node)

print(20*"#")
Tgen.mvPrevNodeContext()
print(Tgen.node)
'''




