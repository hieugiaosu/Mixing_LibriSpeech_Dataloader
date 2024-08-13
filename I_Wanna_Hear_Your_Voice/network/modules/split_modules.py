import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class DimensionDimAttention(nn.Module):
    def __init__(
            self,
            emb_dim: int = 48,
            kernel_size: int = (7,7),
            dilation: int = 2,
            ) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        self.attn = nn.Sequential(
            nn.Conv2d(2*emb_dim,2*emb_dim,1),
            nn.GELU(),
            nn.Conv2d(2*emb_dim,2*emb_dim,kernel_size=kernel_size,groups=2*emb_dim,padding="same"),
            nn.PReLU(),
            nn.Conv2d(2*emb_dim,2*emb_dim,kernel_size=kernel_size,dilation=dilation,groups=2*emb_dim,padding="same"),
            nn.PReLU(),
            nn.Conv2d(2*emb_dim,emb_dim,1),
            nn.Sigmoid()
        )

        self.transform = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            nn.GELU(),
            nn.Conv2d(emb_dim,emb_dim,kernel_size=kernel_size,groups=emb_dim,padding="same"),
            nn.PReLU(),
            nn.Conv2d(emb_dim,emb_dim,kernel_size=kernel_size,dilation=dilation,groups=emb_dim,padding="same"),
            nn.PReLU(),
            nn.Conv2d(emb_dim,emb_dim,1)
        )

    def forward(self,x,e):
        """
        Args:
            x: (B, D, F, T) input tensor from privous layer
            e: (B, D, F) embedding after reshape
        """
        
        T = x.shape[-1]
        emb = repeat(e, 'B D F -> B D F T', T=T)
        att = torch.cat([x,emb],dim=1)

        i = self.transform(x)
        att = self.attn(att)
        return i*att
    
class FDAttention(nn.Module):
    def __init__(
            self
            ) -> None:
        super().__init__()
    def forward(self,x,e):
        """
        Args:
        
        x: (B, D, F, T) input tensor from privous layer (for k and v)
        e: (B, D, F) embedding after reshape (for q)
        """

        _,D,n_freq,T = x.shape
        q = repeat(e, 'B D F -> B T (D F)', T=T)
        k = rearrange(x, 'B D F T -> B (D F) T')
        v = rearrange(x, 'B D F T -> B T (D F)')

        q = self.positional_encoding(q)
        qkT = torch.matmul(q,k)/((D*n_freq)**0.5)
        qkT = F.softmax(qkT,dim=-1)
        att = torch.matmul(qkT,v)
        
        att = rearrange(att, 'B T (D F) -> B D F T', D=D, F=n_freq)
        return att

    def positional_encoding(self, x):
        """
        Args:
            x: (B, T, D) input to add positional encoding
        """
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * (-torch.log(torch.tensor(10000.0)) / D))
        
        pos_enc = torch.zeros_like(x)
        pos_enc[:, :, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, :, 1::2] = torch.cos(pos * div_term)
        return x + pos_enc

class SplitModule(nn.Module):
    def __init__(
            self,
            emb_dim: int = 48,
            condition_dim: int = 256,
            n_fft: int = 128,
            ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.condition_dim = condition_dim
        n_freq = n_fft // 2 + 1
        self.n_freqs = n_freq

        self.alpha = nn.Parameter(torch.empty(1,emb_dim,self.n_freqs).to(torch.float32))
        self.beta = nn.Parameter(torch.empty(1,emb_dim,self.n_freqs,1).to(torch.float32))
        self.d_att = DimensionDimAttention(emb_dim=emb_dim)

        # self.f_att = FDAttention()
    def forward(self,input,emb):
        """
        Args:
            input: (B, D, F, T) input tensor
            emb: (B, D, F) embedding after reshape
        """
        e = F.tanh(emb*self.alpha)

        x = self.d_att(input,e)
        # x = self.f_att(x,e)
        x = x*F.sigmoid(self.beta*emb.unsqueeze(-1))
        return x