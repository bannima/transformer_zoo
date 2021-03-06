#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: example.py
@time: 2022/2/8 10:19 AM
@desc: 
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformer.train import Batch,LabelSmoothing,NoamOpt,run_epoch
from transformer.module import make_model,subsequent_mask

def data_gen(V,batch,nbatches):
    " Generate random data for a src-tgt copy task. "
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1,V,size=(batch,10)))
        data[:,0] = 1
        src = Variable(data,requires_grad=False)
        tgt = Variable(data,requires_grad=False)
        yield Batch(src,tgt,0)

class SimpleLossCompute(object):
    " A simple loss compute and train function. "
    def __init__(self,generator,criterion,opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self,x,y,norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1,x.size(-1)),
                              y.contiguous().view(-1))/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()*norm

if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V,padding_idx=0,smoothing=0.0)
    model = make_model(V,V,N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model,1,400,
                        torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9))
    for epoch in range(11):
        model.train()
        run_epoch(data_gen(V,30,20),model,
                  SimpleLossCompute(model.generator,criterion,model_opt))
        model.eval()
        print(run_epoch(data_gen(V,30,5),model,SimpleLossCompute(model.generator,criterion,None)))

