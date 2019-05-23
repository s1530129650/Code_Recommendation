#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: three_plot.py
@time: 2019/1/14 17:23
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot


'''
#dim = 300 epoch =20
xAxix = np.arange(1,21,1)

epoTrainLoss  = [0.448495, 0.508751, 0.532357, 0.6, 0.613992, 0.610009, 0.643922,
            0.633015, 0.662563, 0.654863,0.754189, 0.773775, 0.790345, 0.766849,
            0.793038, 0.801976,0.798726, 0.777984, 0.779387, 0.779058]
trainAcys =[0.503548, 0.593451, 0.647606, 0.608341, 0.642637, 0.659692, 0.685261, 0.689824,
            0.690533, 0.703364,0.725963, 0.732074, 0.757338, 0.771809, 0.755751, 0.765505,
            0.761222, 0.744889, 0.773701, 0.761178]
testAcys = [0.4625, 0.564775, 0.599277, 0.562068, 0.584479, 0.611812, 0.6303,
            0.640704, 0.635766, 0.654725,0.686102, 0.693176, 0.708484, 0.731966, 0.711915, 0.730567,
            0.719484, 0.700975, 0.720009, 0.721488]
'''

# 140000 60000 100000
xAxix = np.arange(1,11,1)
epoTrainLoss  =[0.19442062096255167, 0.17990774575982774, 0.1731374791775431, 0.16709070628285408, 0.16238965774221079, 0.15870868202745914, 0.15546583086252214, 0.15277694323637656, 0.15047304502044406, 0.14802473999496016]
trainAcys =[0.3151083333333429, 0.3315380555555625, 0.3436008333333392, 0.35304083333333636, 0.35440194444444956, 0.35887833333333713, 0.36405944444444804, 0.36011583333333697, 0.3618941666666712, 0.3628622222222258]
testAcys = [0.5241661111110955, 0.5489549999999841, 0.5698305555555415, 0.5851499999999893, 0.5858197222222106, 0.5993441666666545, 0.6047344444444366, 0.5954488888888786, 0.600843055555548, 0.6035763888888787]

# Plot


host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
par1 = host.twinx()

# set labels
plt.title('Result(dim =300)',fontsize='large',fontweight='bold')
host.set_xlabel("epoch")
host.set_ylabel("log loss")
par1.set_ylabel("accuracy")
#par2.set_ylabel("validation accuracy")


# plot curves
p1, = host.plot(xAxix , epoTrainLoss, label="training log loss")
#p2, = par2.plot(xAxix , trainAcys  , label="validation accuracy")
p2, = par1.plot(xAxix , trainAcys  , label="validation accuracy")
p3, = par1.plot(xAxix , testAcys , label="test accuracy")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)


# set label color
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p3.get_color())
#par2.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([0,12])
host.set_ylim([0.13,0.21])
par1.set_ylim([0.2, 0.7])
#par1.set_ylim([0., 1.05])

plt.draw()
plt.show()
