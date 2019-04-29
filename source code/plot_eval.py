#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: plot_eval.py
@time: 2019/4/28 15:22
"""
train_loss=[0.21523026615311924, 0.20525309194013414, 0.19801966458179832, 0.18867008883415845, 0.17902659684855923]
eval_loss =[0.21097177310538362, 0.2034591461739979, 0.20031716386701057, 0.19787413916420823, 0.20067148275591895]

import tensorflow as tf

from tensorboardX import SummaryWriter
'''
megred = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./log/train', sess.graph)
summary_writer1 = tf.summary.FileWriter('./log/test')
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)
    writer_test = tf.summary.FileWriter(test_log_dir)    # 注意此处不需要sess.graph

    writer_train.add_summary(summary_str_train,step)
    writer_test.add_summary(summary_str_test,step)




writer = SummaryWriter('LeNet5')

writer = SummaryWriter('./log/train')
writer1 = SummaryWriter('./log/test')
for epoch in range(5):
    epoch_loss1 = train_loss[epoch]
    epoch_loss2 = eval_loss[epoch]
    writer.add_scalar('Train/Loss', epoch_loss1, epoch)
    writer1.add_scalar('val/Loss', epoch_loss2, epoch)
writer.close()

writer1.close()
'''
train_log_dir = 'logs/train/'
test_log_dir = 'logs/test/'   # 两者路径不同
megred = tf.summary.merge_all()
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)
    writer_test = tf.summary.FileWriter(test_log_dir)    # 注意此处不需要sess.graph
    for epoch in range(5):
        epoch_loss1 = train_loss[epoch]
        epoch_loss2 = eval_loss[epoch]

        writer_train.add_summary(epoch_loss1, epoch)
        writer_test.add_summary(epoch_loss2, epoch)

