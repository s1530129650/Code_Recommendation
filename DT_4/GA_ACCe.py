#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: GA_ACCe.py
@time: 2019/5/15 9:12
"""


import numpy as np
import geatpy as ga # 导入geatpy库
import time

"""============================函数定义============================"""
# 传入种群染色体矩阵解码后的基因表现型矩阵Phen以及种群个体的可行性列向量LegV
def aimfuc(Phen, LegV):
    x = Phen
    f = x
    return [f, LegV]
"""============================变量设置============================"""
x1 = [0,8]
codes = [1]                     # 变量的编码方式，2个变量均使用格雷编码


ranges=np.vstack(x1 )     # 生成自变量的范围矩阵
borders=np.vstack([1,1])      # 生成自变量的边界矩阵
"""========================遗传算法参数设置========================="""
NIND = 50                # 种群规模
MAXGEN = 10            # 最大遗传代数
GGAP = 0.8               # 代沟：子代与父代个体不相同的概率为0.8
selectStyle = 'sus';     # 遗传算法的选择方式设为"sus"——随机抽样选择
recombinStyle = 'xovdp'  # 遗传算法的重组方式，设为两点交叉
recopt = 0.9             # 交叉概率
pm = 0.1                 # 变异概率
SUBPOP = 1               # 设置种群数为1
maxormin = -1             # 设置最大最小化目标标记为1，表示是最小化目标，-1则表示最大化目标
"""=========================开始遗传算法进化========================"""
FieldD = ga.crtfld(ranges,borders) # 调用函数创建区域描述器
Lind = np.sum(FieldD[0, :]) # 计算编码后的染色体长度
print(FieldD)
Chrom = ga.crtbp(NIND, FieldD ) # 根据区域描述器生成二进制种群
Phen = ga.bs2int(Chrom, FieldD) #对初始种群进行解码
LegV = np.ones((NIND, 1)) # 初始化种群的可行性列向量
[ObjV,LegV] = aimfuc(Phen, LegV) # 计算初始种群个体的目标函数值
# 定义进化记录器，初始值为nan
pop_trace = (np.zeros((MAXGEN, 2)) * np.nan)
# 定义种群最优个体记录器，记录每一代最优个体的染色体，初始值为nan
ind_trace = (np.zeros((MAXGEN, Lind)) * np.nan)
# 开始进化！！
start_time = time.time() # 开始计时
for gen in range(MAXGEN):
    FitnV = ga.ranking(maxormin * ObjV, LegV) # 根据目标函数大小分配适应度值
    SelCh=ga.selecting(selectStyle, Chrom, FitnV, GGAP, SUBPOP) # 选择
    SelCh=ga.recombin(recombinStyle, SelCh, recopt, SUBPOP) #交叉
    SelCh=ga.mutbin(SelCh, pm) # 二进制种群变异
    Phen = ga.bs2rv(SelCh, FieldD) # 对育种种群进行解码(二进制转十进制)
    LegVSel = np.ones((SelCh.shape[0], 1)) # 初始化育种种群的可行性列向量
    [ObjVSel,LegVSel] = aimfuc(Phen, LegVSel) # 求育种个体的目标函数值
    [Chrom,ObjV,LegV] = ga.reins(Chrom,SelCh,SUBPOP,1,1,maxormin*ObjV,maxormin*ObjVSel
    ,ObjV,ObjVSel,LegV,LegVSel) # 重插入得到新一代种群
    # 记录
    pop_trace[gen, 1] = np.sum(ObjV) / ObjV.shape[0] # 记录当代种群的目标函数均值
    if maxormin == 1:
        best_ind = np.argmin(ObjV) # 计算当代最优个体的序号
    elif maxormin == -1:
        best_ind = np.argmax(ObjV)
    pop_trace[gen, 0] = ObjV[best_ind] # 记录当代种群最优个体目标函数值
    ind_trace[gen, :] = Chrom[best_ind, :] # 记录当代种群最优个体的变量值
# 进化完成
end_time = time.time() # 结束计时
"""============================绘图================================"""
ga.trcplot(pop_trace, [['最优个体目标函数值','种群的目标函数均值']], ['demo_result'])

"""============================输出结果============================"""
best_gen = np.argmin(pop_trace[:, 0]) # 计算最优种群是在哪一代
print('最优的目标函数值为：', np.min(pop_trace[:, 0]))
print('最优的控制变量值为：')
# 最优个体记录器存储的是各代种群最优个体的染色体，此处需要解码得到对应的基因表现型
variables = ga.bs2rv(ind_trace, FieldD) # 解码
for i in range(variables.shape[1]):
    print(variables[best_gen, i])
print('最优的一代是第',best_gen + 1,'代')
print('用时：', end_time - start_time, '秒')