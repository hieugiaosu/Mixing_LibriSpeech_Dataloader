import torch.nn as nn
from einops import rearrange

class FiLMLayer(nn.Module):
    def __init__(self,channels,conditional_dim=256, apply_dim = 1):
        super().__init__()
        self.alpha = nn.Linear(conditional_dim,channels)
        self.beta = nn.Linear(conditional_dim,channels)
        self.apply_dim = apply_dim
    def forward(self,x,condition):
        alpha = self.alpha(condition)
        beta = self.beta(condition)
        input = x
        if self.apply_dim != 1:
            input = input.transpose(1,-1)
        alpha = rearrange(alpha,"b d -> b d"+" 1"*(x.dim()-alpha.dim()))
        beta = rearrange(beta,"b d -> b d"+" 1"*(x.dim()-beta.dim()))
        out = alpha*input+beta
        if self.apply_dim != 1:
            out = out.transpose(1,-1)
        return out