#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: module.py
@time: 2022/2/7 1:43 PM
@desc: Implementation of transformer, code are from `The Annotated Transformer`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

def subsequent_mask(size):
    " Mask out subsequent positions "
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0

def clones(module,N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
    " Compute `Scaled Dot Porduct Attention`"
    d_k = query.size(-1) #dimension of queries and keys
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9) #fill the mask place with -inf
    p_attn = F.softmax(scores,dim=-1) # softmax ,note that exp(-inf)=0
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn

class LayerNorm(nn.Module):
    " Construct a layernorm module "
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderDecoder(nn.Module):
    """ A standard Encoder-Decoder architecture. """
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        " Take in and process masked src and target sequences. "
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

class Encoder(nn.Module):
    """ Core encoder is a stack of N layers """
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        """ Pass the input (and mask) through each layer in turn. """
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    " A residual connection followed by a layer norm."
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        " Apply residual connection to any sublayer with the same size. "
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    " Encoder is made up of self-attn and feed forward "
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size

    def forward(self,x,mask):
        x  = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

class Decoder(nn.Module):
    " Generic N layer decoder with masking "
    def __init__(self,layer,N):
        super(Decoder,self).__init__()

        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    " Decoder is made of self-attn, src-attn, and feed forward "
    def __init__(self, size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)

class MultiHeadAttention(nn.Module):
    " Implementation of multi-head attention "
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h ==0
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query,key,value = [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
                           for l,x in zip(self.linears,(query,key,value))]

        # 2) Apply attention on all the projected vectors in batch.
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

        # 3) "Concat " using a view and apply a final linear
        x = x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedFoward(nn.Module):
    " Implements the FeedForwardNetwork"
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedFoward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embedddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embedddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model

    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    " Implement the Positional Encoding func "
    def __init__(self,d_model,dropout,max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # compute the positional encodings aonce in log space.
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe) # register buffer for variable `pe`

    def forward(self,x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

# Full Model
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    " Construct a model from hyperparaneters "
    c = copy.deepcopy
    attn = MultiHeadAttention(h,d_model)
    ff = PositionwiseFeedFoward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        decoder=Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        src_embed = nn.Sequential(Embedddings(d_model,src_vocab),c(position)),
        tgt_embed = nn.Sequential(Embedddings(d_model,tgt_vocab),c(position)),
        generator = Generator(d_model,tgt_vocab)
    )

    #initialize parameters with glorot
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model

if __name__ == '__main__':
    plt.figure(figsize=(5,5))
    mask = subsequent_mask(10)
    plt.imshow(mask[0])
    plt.show()

    # plt.figure(figsize=(15,5))
    # pe = PositionalEncoding(20,0)
    # y = pe.forward(Variable(torch.zeros(1,100,20)))
    # plt.plot(np.arange(100),y[0,:,4:8].data.numpy())
    # plt.legend(["dim %d"%p for p in [4,5,6,7]])
    # plt.show()

    model_demo = make_model(10,10,2)














