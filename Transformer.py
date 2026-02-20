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


        if mask is not None: # mask should be like this - [[1,0,0],[1,1,0],[1,1,1]]
            score = score.masked_fill(mask == 0, -1e9)  # replace with -1e9 wherever condition is True.

        score=score.softmax(dim=-1) # For each fixed (batch, row), apply softmax across columns.
        # (batch , seq_len, seq_len)
        """When you apply Softmax along a specific dimension, you are telling the computer: "Look at the numbers along this axis and turn them into probabilities that add up to 1.0."""

        if dropout is not None:
            score = dropout(score)
        
        return score@v, score # returning score just in case if have to visualize 


    def forward(self,q,k,v,mask=None): # mask : (Batch, Seq_Len, Seq_Len)
        query=self.Wq(q) # (batch , seq_len, d_model)
        key=self.Wk(k)  # (batch , seq_len, d_model)
        value=self.Wv(v)  #(batch , seq_len, d_model)

        # now have to split into 8 heads

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)   # (batch, h, seq_len, d_k)
        # why not key.view(batch, h, seq_len, d_k) directly 
        # coz hum randomly swap nhi kr skte keep first two same will allow us to divide the 512 in diff heads
        # We cannot randomly swap dimensions using view() because view() does not change memory layout. We must first split the embedding dimension (which is contiguous), then use transpose() to move the head dimension forward.
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) #   (batch, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.scaled_dot_product_attention(query, key, value, self.dropout,mask)
        # reshape # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) 

        return self.Wo(x)



class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter # 512
        self.bias = nn.Parameter(torch.zeros(d_model)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1) # this mean for each word we have a mean
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardNN(nn.Module):
    def __init__(self,d_model,d_ffn,dropout):
        super().__init__()

        self.fnn=nn.Sequential(
            nn.Linear(d_model,d_ffn), #  (batch, seq_len, d_model)--> (batch, seq_len, d_ffn)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn,d_model) # (batch, seq_len, d_ffn) --> (batch, seq_len, d_model)
        )

    def forward(self, x):
        return self.fnn(x)



class EncoderLayer(nn.Module):
    def __init__(self,h:int,d_fnn:int, d_model:int , dropout:float):
        super().__init__()

        self.attention=MultiHeadAttention(h=h,dropout=dropout,d_model=d_model)# h:int,dropout,d_model:int
        self.norm1=LayerNormalization(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=FeedForwardNN(d_model,d_ffn=d_fnn,dropout=dropout)
        self.norm2=LayerNormalization(d_model)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        residual_1=x
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x) # This can be ignored coz we have already applied dropout to each head  
        x=self.norm1(x+residual_1)

        residual_2=x    
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+residual_2)

        return x
    
class Encoder(nn.Module):
    def __init__(self,n_layer,h:int,d_fnn:int, d_model:int , dropout:float):
        super().__init__()
        # can't or should use nn.Sequential coz it only pass one argument and will not pass mask 
        self.layers=nn.ModuleList([EncoderLayer(h=h,d_fnn=d_fnn,d_model=d_model,dropout=dropout) for _ in range(n_layer)])

    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        return x




## DECODER



class DecoderLayer(nn.Module):
    def __init__(self,h,dropout,d_model,d_ffn):
        super().__init__()
        self.masked_attention=MultiHeadAttention(h,dropout,d_model)
        self.norm1=LayerNormalization(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.cross_attention=MultiHeadAttention(h,dropout,d_model)
        self.dropout2=nn.Dropout(dropout)
        self.norm2=LayerNormalization(d_model)

        self.ffn=FeedForwardNN(d_model,d_ffn,dropout)
        self.norm3=LayerNormalization(d_model)
        self.dropout3=nn.Dropout(dropout)

    def forward(self,x,y,src_mask,tgt_mask):
        residual_1=y
        y=self.masked_attention(y,y,y,tgt_mask)
        y=self.dropout1(y)
        y=self.norm1(y+residual_1)
        residual_2=y
 
        y=self.cross_attention(y,x,x,src_mask)
        y=self.dropout2(y)
        y=self.norm2(y+residual_2)

        residual_3=y
        y=self.ffn(y)
        y=self.dropout3(y)
        y=self.norm3(y+residual_3)

        return y


class Decoder(nn.Module):
    def __init__(self,n_layer:int,h,dropout,d_model,d_ffn):
        super().__init__()
        self.layers=nn.ModuleList([DecoderLayer(h,dropout,d_model,d_ffn) for _ in range(n_layer)])

    def forward(self,x,y,src_mask, tgt_mask):
        for layer in self.layers:
            y=layer(y,x,src_mask, tgt_mask)
        return y

        




