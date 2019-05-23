#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: v-enshi
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: GA_1.py
@time: 2019/5/14 17:31
"""

import numpy as np
import matplotlib.pyplot as plt
from math import log2,log
import copy
from collections import Counter
from numba import jit

import time



def GA_Algorithm(baseEntropy ,query_data,instr):
    feature = np.arange(0,len(instr))
    len_fea = len(feature)
    DNA_SIZE = int(log2(len_fea )) + 1         # DNA length
    POP_SIZE = 4                         # population size
    CXPB = 0.5
    MUTATION_RATE = 0.002                    # mutation probability
    N_GENERATIONS = 3
    min_size = 500
    #print("POP_SIZE ",POP_SIZE," N_GENERATIONS", N_GENERATIONS)
    # 2 calulate entropy
    def cal_entropy(data_cal):
        label_list = []
        numEntries = len(data_cal)

        for i in range(numEntries):
            label_list.append(data_cal[i].label)
        labelCounts = Counter(label_list)
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    # 4 split data
    def split_set(universalSet, item):
        retDataSet = []
        i = 0
        #print("item",item)
        while i < (len(universalSet)):
            if universalSet[i].ctx[-1] == item:  #

                retDataSet.append(universalSet[i])
                universalSet.remove(universalSet[i])
                i = i - 1
            i = i + 1
        # print("split",len(retDataSet))
        #print("dataSet_ch", len(universalSet))
        return retDataSet

    def F(pop_x):

        infoGainRatios = []
        for  x in pop_x:
            Pfeature = instr[x]

            dataSet_ch = copy.deepcopy(query_data)
            length_data = len( dataSet_ch)
            print("start count label")
            fea_list = []
            for data_raw in dataSet_ch:
                data_raw.geneProgram(Pfeature)
                fea_list.append(data_raw.ctx[-1])

            uniqueCtx_ch = Counter(fea_list)
            uniqueCtx_ch  = uniqueCtx_ch.most_common(10)


            if (not uniqueCtx_ch[0][1] < length_data)  or uniqueCtx_ch[0][1] < min_size \
                    or uniqueCtx_ch[0][0] == "EMPTY" or uniqueCtx_ch[0][0] == "need_to_predict" :
                infoGainRatios.append(-1000)
                continue


            for i in range(len(uniqueCtx_ch)):
                #print( uniqueCtx_ch[i] )
                if uniqueCtx_ch[i][1] < min_size:
                    del uniqueCtx_ch[i:]
                    break
            print(" uniqueCtx_ch", uniqueCtx_ch)
            newEntropy = 0.0
            splitInfo = 0.0

            print("start calculate IGR")
            for last_ctx in uniqueCtx_ch:
                subDataSet = split_set(dataSet_ch, last_ctx[0])  #
                #print("dataSet_ch",len(dataSet_ch))
                prob = len(subDataSet) / float( length_data)
                newEntropy += prob * cal_entropy(subDataSet)
                if prob == 0:
                    splitInfo += 0
                else:
                    splitInfo += -prob * log(prob, 2)
            if len(dataSet_ch) > 0:
                prob = len(dataSet_ch) / float(length_data)
                newEntropy += prob * cal_entropy(dataSet_ch)
                if prob == 0:
                    splitInfo += 0
                else:
                    splitInfo += -prob * log(prob, 2)
            #now4 = time.time()
            #print(" calcukate IGR", now3 - now4)
            infoGain = baseEntropy - newEntropy  # feature  infoGain
            if splitInfo == 0:  # fix the overflow bug
                infoGainRatio = 0
            else:
                infoGainRatio = infoGain / splitInfo  # feature infoGainRatio
            infoGainRatios.append(infoGainRatio)
        return np.array(infoGainRatios)


        return x      # to find the maximum of this function

    # find non-zero fitness for selection
    def get_fitness(pred): return pred + 1e-4 - np.min(pred)

    # convert binary DNA to decimal and normalize it to a range(0, 5)
    def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) % len_fea


    def select(pop, fitness,num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        #print("len fit",len(fitness))
        idx = np.random.choice(POP_SIZE, size=num_parents, replace=False,p=fitness/fitness.sum())
        pars = pop[idx]
        pars_fi = fitness[idx]
        return pars,pars_fi


    def crossover(parents,parents_fit , offspring_size):     # mating process (genes crossover)

        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.

        len_par = len(parents)
        for k in range(offspring_size[0]):
            crossover_point = np.random.randint(offspring_size[1])
            parent_idx = np.random.choice(len_par, size=2, replace=False,
                                          p=parents_fit / parents_fit.sum())

            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent_idx[0], 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent_idx[1], crossover_point:]
        return offspring



    def mutate(children):

        for child in children:
            point = np.random.randint(DNA_SIZE)
            if np.random.rand() < MUTATION_RATE:
                child[point] = 1 if child[point] == 0 else 0

        return children


    new_pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA
    #print("new_pop len",len(new_pop))

    for _ in range(N_GENERATIONS):
        # Measuring the fitness of each chromosome in the population.

        #now1 = time.time()
        F_values = F(translateDNA(new_pop))    # compute function value by extracting DNA
        #now2 = time.time()
        #print("F",now2-now1)

        # GA part (evolution)

        fitness = get_fitness(F_values)
        #print("Most fitted DNA: ", new_pop[np.argmax(fitness), :],translateDNA(new_pop[np.argmax(fitness), :]))
        # Selecting the best parents in the population for mating.


        parents,parents_fit = select(new_pop, fitness,int(POP_SIZE/2) )


        # Generating next generation using crossover.
        # Generating next generation using crossover.

        offspring_crossover = crossover(parents,parents_fit ,
                                           offspring_size=(POP_SIZE - parents.shape[0], DNA_SIZE))

        # Adding some variations to the offsrping using mutation.

        offspring_mutation = mutate(offspring_crossover)
        # Creating the new population based on the parents and offspring.
        new_pop[0:parents.shape[0], :] = parents
        new_pop[parents.shape[0]:, :] = offspring_mutation

    return translateDNA(new_pop[np.argmax(fitness), :])
