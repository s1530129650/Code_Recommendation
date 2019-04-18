#!/usr/bin/env python
#!-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence 
@contact: 123@qq.com
@site: 
@software: PyCharm
@file: pointer_atten.py
@time: 2019/4/9 15:51
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dfs import dfs_AST
import json
import random

import time
time_start=time.time()
USE_CUDA = False


# =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"


torch.manual_seed(1)
'''
EMBEDDING_value = 1200
EMBEDDING_type = 300
HIDDEN_SIZE = 1500
CONTEXT_WINDOW = 50
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 50
max_vocab_size = 1000
'''
EMBEDDING_value = 2
EMBEDDING_type = 3
HIDDEN_SIZE = 5
CONTEXT_WINDOW = 3
#BATCH_SIZE = 2
BATCH_SIZE = 1
context_window = 3
max_vocab_size = 4

data = [[ {"type":"Module","children":[1,4]},{"type":"Assign","children":[2,3]},  {"type":"NameStore","value":"x"},
         {"type":"Num","value":"7"},{"type":"Print","children":[5]},{"type":"BinOpAdd","children":[6,7]},
         {"type":"NameLoad","value":"x"}, {"type":"Num","value":"1"} ]]

#str =r"D:\v-enshi\Language model\suggestion\Code Completion with Neural Attention and Pointer Networks"
# data losding
'''
data = []
with open("../data/python/python100k_train.json",'r') as load_f:
    data1 = load_f.readlines()
for i in range(len(data1)):
    content = json.loads(data1[i])
    data.append(content)
'''


data_flatten = []
for i  in range(len(data)):
    data_flatten .append(dfs_AST(data[i], 0))


#print("data_flatten:",data_flatten)

training_data= []
for datalist in data_flatten:
    inner_data = []
    for items in datalist:
        inner_data.append({items["type"]:items["value"]})
    training_data.append(inner_data)
#print("training_data:",training_data)


# build vocabulary
type_to_ix = {"EOF":0}
word_to_ix = {}
for i in range(len(training_data)):
    for it in training_data[i]:
        for type , value in it.items():
            if type not in type_to_ix:
                type_to_ix[type] = len(type_to_ix)
            if value in word_to_ix:
                word_to_ix[value] = word_to_ix[value] + 1
            else:
                word_to_ix[value] = 1


#print("word_to_ix",word_to_ix)
#print("type_to_ix",type_to_ix)
L = sorted(word_to_ix.items(),key=lambda item:item[1],reverse=True)
value_to_ix = {"UNK":0,"EOF":1}
for i in range(max_vocab_size ):
    value_to_ix [L[i][0]] = len(value_to_ix )



# random move a node
training_data_rd = []
vocab_size = len(value_to_ix )
for i in range(len(training_data)):
    length = len(training_data[i])
    if length <= CONTEXT_WINDOW+1:
        continue
    rd = random.randint(CONTEXT_WINDOW + 1, len(training_data[i]) - 1)
    inner_data = [training_data[i][:rd], [training_data[i][rd]], -1, rd] #[context,global position, context position, self -position)
    for j in range(rd-1 , rd- CONTEXT_WINDOW,-1): #whether the remove node in the context.if the node in context,we remeber the position in context
        if j < 0:
            break  #some sequence length is less than CONTEXT_WINDOW
        if training_data[i][rd] == training_data[i][j]:
            inner_data = [training_data[i][:rd], [training_data[i][rd]], vocab_size+rd - j -1, rd]
            break
    training_data_rd.append(inner_data)
#print(training_data_rd)


# pad the EOF
for i in range(len(training_data_rd)):
    if not len(training_data_rd[i][0]) % context_window == 0:
        pad = {"EOF":"EOF"}
        for j in range(context_window - len(training_data_rd[i][0]) % context_window):
            training_data_rd[i][0].append(pad)
#print("training_data:",training_data_rd)
print("training data size:",len(training_data_rd))
#print(training_data_rd)


class LSTM_component(nn.Module):

    def __init__(self,vocab_value_size, value_dim,vocab_type_size, type_dim,
                 hidden_dim, batch_size):
        super(LSTM_component, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.value_embeddings = nn.Embedding(vocab_value_size, value_dim)
        self.type_embeddings = nn.Embedding(vocab_type_size, type_dim)
        self.lstm = nn.LSTM(value_dim + type_dim, hidden_dim)

        self.value_embeddings = self.value_embeddings.cuda() if USE_CUDA else self.value_embeddings
        self.type_embeddings = self.type_embeddings.cuda() if USE_CUDA else self.type_embeddings
        self.lstm = self.lstm.cuda()  if USE_CUDA else self.lstm


    def forward(self,sentence,hc):
        #senten=[[1 2 3],[4  5 6]] sentence[0]is value  index ..[1] is type index
        embeds_value = self.value_embeddings(sentence[0])
        embeds_value =  embeds_value.cuda() if USE_CUDA else  embeds_value

        embeds_type  = self.type_embeddings(sentence[1])
        embeds_type = embeds_type.cuda() if USE_CUDA else  embeds_type

        embeds = torch.cat([embeds_value,embeds_type],1).view(len(sentence[0]),1,-1)
        embeds = embeds.cuda() if USE_CUDA else  embeds

        h0 = hc[0]
        c0 = hc[1]
        lstm_out, (lstm_h,lstm_c) = self.lstm(embeds,(h0,c0))

        lstm_out = lstm_out.cuda() if USE_CUDA else lstm_out
        lstm_h = lstm_h.cuda()if USE_CUDA else lstm_h
        lstm_c = lstm_c.cuda()if USE_CUDA else lstm_c


        return  lstm_out ,lstm_h,lstm_c



#model_LSTM = LSTM_component(len(value_to_ix ),EMBEDDING_value,len(type_to_ix),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE)



def prepare_sequence(seq,val_to_ix,ty_to_ix):# trans code to idex
    idxs_ty = []
    idxs_vl = []
    for w in seq:
        for k,v in w.items():
            if v in val_to_ix.keys():
                idxs_vl.append(val_to_ix[v])
            else:
                idxs_vl.append(val_to_ix["UNK"])
            idxs_ty.append(ty_to_ix[k])
    output = torch.tensor([idxs_vl ,idxs_ty ],dtype = torch.long)
    return output
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data_rd[0][0],value_to_ix ,type_to_ix)
    print(inputs)
    output, hn,cn = model_LSTM(inputs,model_LSTM.initHidden())
    print("tag_scorces", output, hn,cn)
'''

class Context_atten(nn.Module):
    def __init__(self, hidden_dim,context_window):
        super(Context_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.Wm = nn.Parameter(torch.ones(hidden_dim,hidden_dim))
        self.V = nn.Parameter(torch.ones(hidden_dim, 1))
        self.linear1 = nn.Linear(hidden_dim,hidden_dim,bias= False)


        self.linear1 = self.linear1.cuda() if USE_CUDA else self.linear1



    def forward(self, inputs, hc):


        Mt = inputs[-self.context_window-1:-1,:,:]
        Mt = Mt.view(self.context_window, self.hidden_dim) # 3*K


        one_TL = torch.ones(context_window,1) #(L,1)
        one_TL = one_TL.cuda() if USE_CUDA else one_TL

        At = torch.mm(torch.tanh(torch.mm(Mt,self.Wm) + torch.mm(one_TL, self.linear1(hc.view(1, -1)))) , self.V)
        At = At.cuda() if USE_CUDA else At

        alphat =F.log_softmax(At.view(1,-1),dim = 1) # [1,3]
        alphat = alpha.cuda() if USE_CUDA else alphat

        ct = torch.mm(alphat,Mt)
        ct = ct.cuda() if USE_CUDA else ct

        return  alphat,ct


#model_catten = Context_atten(HIDDEN_SIZE, CONTEXT_WINDOW)
'''
with torch.no_grad():
    alpha , out_ct = model_catten (output, hn)
    print(out_ct)  
'''

class Parent_atten(nn.Module):
    def __init__(self,hidden_dim,vocab_size):
        super(Parent_atten,self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.Wg_linear = nn.Linear(hidden_dim*2,hidden_dim,bias= False).cuda() if USE_CUDA else nn.Linear(hidden_dim*2,hidden_dim,bias= False)
        self.Wv_linear = nn.Linear(hidden_dim , vocab_size).cuda()if USE_CUDA else nn.Linear(hidden_dim , vocab_size)

        self.Wg_linear = self.Wg_linear.cuda() if USE_CUDA else self.Wg_linear
        self.Wv_linear = self.Wv_linear.cuda() if USE_CUDA else self.Wv_linear

    def forward(self,ht,ct):

        #Gt = torch.tanh(self.Wg_linear( torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        Gt = torch.tanh(self.Wg_linear( torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        Gt = Gt.cuda() if USE_CUDA else Gt

        yt = F.log_softmax(self.Wv_linear(Gt),dim= 1)
        yt = yt.cuda() if USE_CUDA else yt

        return yt

#model_par_atten_type = Parent_atten(HIDDEN_SIZE, len(type_to_ix))
'''
with torch.no_grad():
    Yt_type = model_par_atten_type  ( hn,out_ct)
    print(Yt_type)
'''


#model_par_atten_value = Parent_atten(HIDDEN_SIZE, len(value_to_ix ))

'''
with torch.no_grad():
    Yt_value = model_par_atten_value ( hn,out_ct)
    print(Yt_value)
'''


class Point_Mixture(nn.Module):
    def __init__(self,hidden_dim):
        super(Point_Mixture, self).__init__()
        self.Ws_linear = nn.Linear(hidden_dim*2,1)

        self.Ws_linear =  self.Ws_linear.cuda()  if USE_CUDA else self.Ws_linear

    def forward(self, ht,ct,wt,lt):

        st = torch.sigmoid(self.Ws_linear(torch.cat([ht.view(1,-1),ct.view(1,-1)],1)))
        st = st.cuda() if USE_CUDA else st

        yt = torch.cat( [torch.mm(st,wt.view(1,-1)) , torch.mm(1 - st,lt.view(1,-1))] ,1)
        yt = yt.cuda() if USE_CUDA else yt

        #model_par_atten_value = Parent_atten(HIDDEN_SIZE, len(value_to_ix ))


        return yt


#model_Point_Mix = Point_Mixture(HIDDEN_SIZE)
'''
with torch.no_grad():
    yt_point = model_Point_Mix(hn,out_ct, Yt_value,alpha )
    print(yt_point)

'''
#loss_function = nn.NLLLoss()

class main_model(nn.Module):
    def __init__(self,vocab_value_size, value_dim,vocab_type_size, type_dim,
                 hidden_dim, batch_size,context_window):
        super(main_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.model_LSTM = LSTM_component(vocab_value_size, value_dim, vocab_type_size, type_dim, hidden_dim, batch_size)
        self.model_catten = Context_atten(hidden_dim, context_window)
        self.model_par_atten_value = Parent_atten(hidden_dim, vocab_value_size)
        self.model_Point_Mix = Point_Mixture(hidden_dim)

        self.model_LSTM = self.model_LSTM.cuda() if USE_CUDA else self.model_LSTM
        self.model_catten = self.model_catten.cuda() if USE_CUDA else self.model_catten
        self.model_par_atten_value = self.model_par_atten_value.cuda() if USE_CUDA else self.model_par_atten_value
        self.model_Point_Mix = self.model_Point_Mix.cuda() if USE_CUDA else  self.model_Point_Mix

    def forward(self,inputs,hc):
        output, hn, cn = self.model_LSTM(inputs,hc)
        output = output.cuda() if USE_CUDA else output
        hn = hn.cuda()  if USE_CUDA else hn
        cn = cn.cuda()  if USE_CUDA else cn

        alpha, out_ct = self.model_catten(output, hn)
        alpha = alpha.cuda() if USE_CUDA else alpha
        out_ct = out_ct.cuda() if USE_CUDA else out_ct


        Yt_value = self.model_par_atten_value(hn, out_ct)
        Yt_value = Yt_value.cuda() if USE_CUDA else Yt_value


        yt_point = self.model_Point_Mix(hn, out_ct, Yt_value, alpha)
        yt_point = yt_point.cuda() if USE_CUDA else yt_point

        return yt_point

    def initHidden(self):
        h0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        h0 = h0.cuda() if USE_CUDA else h0
        c0 = c0.cuda() if USE_CUDA else c0

        return h0, c0


model = main_model(len(value_to_ix ),EMBEDDING_value,len(type_to_ix),EMBEDDING_type,HIDDEN_SIZE, BATCH_SIZE,CONTEXT_WINDOW)
loss_function = nn.NLLLoss()

learning_rate = 0.001
decay = 0.6
optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=decay)

clip = 5
nn.utils.clip_grad_norm_(model.parameters(), clip)


'''
with torch.no_grad():
    inputs = prepare_sequence(training_data_rd[0][0],value_to_ix ,type_to_ix)
    print("inputs",inputs)
    output = model(inputs,model.initHidden())
    print("tag_scorces", output)
'''

losses = []
for epoch in range(8):
    total_loss = 0

    for i in range(len(training_data_rd)):

        sequence = training_data_rd[i][0]
        global_tags = training_data_rd[i][1]
        local_tags  = training_data_rd[i][2]
        targets = prepare_sequence(global_tags, value_to_ix, type_to_ix)
        if local_tags == -1 and targets[0] == 0:
            continue
        optimizer.zero_grad()
        #step 2 prepare the data
        inputs = prepare_sequence(sequence, value_to_ix , type_to_ix)


        if not local_tags == -1 :
            targets[0] = local_tags
        # step1 init

        #step 3 get the scorece
        inputs = inputs.cuda() if USE_CUDA else inouts

        yt_point = model(inputs, model.initHidden())
        #targets[0] = targets[0].cuda  if USE_CUDA else targets[0]
        #step 4 train
        #loss = loss_function(yt_point.view(-1),targets[0])
        loss = loss_function(yt_point.view(1, -1), targets[0])
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
print(losses)

time_end=time.time()

print('totally cost',time_end-time_start)
'''
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
'''
torch.save(model.state_dict(), 'params1w.pkl')
model.load_state_dict(torch.load('params1w.pkl'))

'''
with torch.no_grad():

    inputs = prepare_sequence(training_data[0][0],word_to_ix,type_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
    '''
