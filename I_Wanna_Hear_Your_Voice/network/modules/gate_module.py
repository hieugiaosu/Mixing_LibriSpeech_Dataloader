import torch.nn as nn
import torch
import torch.nn.functional as F

class BandFilterGate(nn.Module):
    def __init__(self,emb_dim=48, n_freqs = 65):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        self.beta = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        nn.init.xavier_normal_(self.alpha)
        nn.init.xavier_normal_(self.beta)
    def forward(self,input,filters,bias):
        f = F.sigmoid(self.alpha*filters)
        b = F.tanh(self.beta*bias)
        print(1111)
        print(input.shape)
        print(f.shape)
        print(b.shape)
        print(2222)
        return f*input + b
        
