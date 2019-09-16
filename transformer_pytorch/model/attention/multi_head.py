#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:41:48 2019

@author: allen
"""
import torch.nn as nn
from .single import attention


class MultiHeadedAttention(nn.Module):
    
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h ==0
        
        # we assume d_v always equal d_k
        self.d_k = d_model // h
        self.h = h
        
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        context = self.output_linear(x)
        return context
