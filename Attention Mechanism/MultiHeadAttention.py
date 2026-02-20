import torch
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    def __init__(self,h:int,dropout,d_model:int):
        super().__init__()
        self.h=h   #8 
        self.d_model=d_model # 512
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head # 64

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo= nn.Linear(d_model,d_model,bias=False)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def scaled_dot_product_attention(q,k,v,dropout,mask=None):
        # q,k shape - (batch,seq_len,d_k)
        d_k=q.shape[-1]
        score=(q@k.transpose(-2,-1))/math.sqrt(d_k) # (batch,seq_len,seq_len)
        if mask is not None: 
            score = score.masked_fill(mask == 0, -1e9)  # replace with -1e9 wherever condition is True.

        score=score.softmax(dim=-1) # For each fixed (batch, row), apply softmax across columns.
        # (batch , seq_len, seq_len)
        if dropout is not None:
            score = dropout(score)
        return score@v, score #returning score just in case if have to visualize 


    def forward(self,q,k,v,mask=None):
        query=self.Wq(q) # (batch , seq_len, d_model)
        key=self.Wk(k)  # (batch , seq_len, d_model)
        value=self.Wv(v)  #(batch , seq_len, d_model)
        
        # now have to split into 8 heads
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)   # (batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) #   (batch, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.scaled_dot_product_attention(query, key, value, self.dropout,mask)
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) 

        return self.Wo(x)














        
