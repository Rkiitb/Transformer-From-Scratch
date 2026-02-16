import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size=vocab_size
        # Standard embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x): # x would be tensor of token ids 
        # According to the paper, we scale the embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

# self attention was called Scaled Dot-Product Attention in original paper
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        # x is the scaled embedding output
        Q = self.Wq(x) 
        K = self.Wk(x)
        V = self.Wv(x)
        
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        
        return torch.matmul(weights, V)
    
