import torch.nn as nn
import torch

class BandFilterGate(nn.Module):
    def __init__(self,emb_dim=48, n_freqs = 65):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        self.beta = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        nn.init.xavier_normal_(self.alpha)
        nn.init.xavier_normal_(self.beta)
    def forward(self,input,filters,bias):
        f = self.alpha*filters
        b = self.beta*bias
        return f*input + b
        
