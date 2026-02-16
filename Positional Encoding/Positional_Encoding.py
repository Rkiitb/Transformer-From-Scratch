import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len,dropout): 
        super().__init__()
        self.d_model=d_model
        self.seq_len=max_seq_len
        self.dropout=nn.Dropout(dropout)


        pe=torch.zero(self.seq_len,self.d_model) 
        pos=torch.arange(0,self.seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # sin - for even pos
        pe[:, 0::2] = torch.sin(pos * div_term)
        # for cos
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Add a batch dimension (1, max_seq_len, d_model) for PyTorch compatibility
        pe = pe.unsqueeze(0)
        # buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

