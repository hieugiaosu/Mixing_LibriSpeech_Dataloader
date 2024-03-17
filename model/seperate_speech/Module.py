import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchaudio.transforms as T 
from einops import rearrange

class DualBlock(nn.Module):

    ## Input for this block will have the shape (B,S,L)
    def __init__(self,embeding_dim,hidden_k,window_size,stack=4 ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if embeding_dim % 2 == 0:
            self.padding = embeding_dim//2
            self.crop = True
        else: 
            self.padding = (embeding_dim-1)//2
            self.crop = False
        self.hidden_k = hidden_k
        self.window_size = window_size
        self.stack = stack
        self.emb_i_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
        self.emb_o_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
        self.positiveLine = nn.ModuleList([
            nn.Conv1d(1,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same")
        ])
        self.negativeLine = nn.ModuleList([
            nn.Conv1d(1,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same")
        ])
        self.bnormP = nn.BatchNorm1d(hidden_k)
        self.bnormN = nn.BatchNorm1d(hidden_k)
    def forward(self,x,e,reshape_input = True,reshape_output=True,batch = None):
        if (not reshape_input) and reshape_output:
            assert batch is not None, "we need batch to reshape"
        if reshape_input:
            batch = x.shape[0]
            i = rearrange(x,"b s l -> (b s) l")
        else: 
            i = x 
        i = i.unsqueeze(0)
        ## i now (1,b*s,l)
        e_i = self.emb_i_trans(e.unsqueeze(1))
        ### now e_i will have shape (B*S,1,E)
        i = F.conv1d(i,e_i,padding = self.padding,groups=i.shape[1]).squeeze()
        i = i.unsqueeze(1)
        if self.crop:
            i = i[:,:,:-1]
        i = F.tanh(i)
        ## now i will have shape (B*S,1,L) or (B*S,1,L+1)
        
        p_i = F.relu(i)
        n_i = -F.relu(-i)
        ## positive flow:
        p_1 = F.tanh(self.positiveLine[0](p_i))
        p_2 = F.tanh(self.positiveLine[1](p_1))
        p_3 = F.tanh(self.positiveLine[2](p_2+p_i))
        p_o = self.bnormP(p_3+p_1)
        ## negative flow:
        n_1 = F.tanh(self.negativeLine[0](n_i))
        n_2 = F.tanh(self.negativeLine[1](n_1))
        n_3 = F.tanh(self.negativeLine[2](n_2+n_i))
        n_o = self.bnormN(n_3+n_1)

        o = n_o + p_o 
        ##o have shape (B*S,hidden_k,L)
        o = o.mean(dim=1)
        o = o.unsqueeze(0)
        e_o = self.emb_o_trans(e.unsqueeze(1))
        o = F.conv1d(o,e_o,padding = self.padding,groups=o.shape[1]).squeeze()
        o = o.unsqueeze(1)
        if self.crop:
            o = o[:,:,:-1]
        o = F.tanh(o)
        
        o = o.squeeze()
        if reshape_output:
            o = rearrange(o,"(b s) l -> b s l",b=batch)
        return o

class DualBlockBaseModel(nn.Module):
    def __init__(self,depth ,embeding_dim,hidden_k,window_size,stack=4,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.ModuleList([DualBlock(embeding_dim,hidden_k,window_size,stack)]*depth)
        if embeding_dim % 2 == 0:
            self.padding = embeding_dim//2
            self.crop = True
        else: 
            self.padding = (embeding_dim-1)//2
            self.crop = False
        self.emb_i_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
    def forward(self,x,e):
        ## x is in shape (B,L), e in shape (B*S,E)
        batch = x.shape[0]
        i = x.unsqueeze(0)
        e_i = self.emb_i_trans(e.unsqueeze(1))

        i = F.conv1d(i,e_i,padding = self.padding,groups=batch)
        if self.crop:
            i = i[:,:,:-1]
        ## now i has shape (1,B*S,L)
        l_i = i.squeeze()
        l_o = 0
        for idx in range(len(self.block)):
            l_i = F.tanh(l_i + l_o)
            l_o = self.block[idx](l_i,e,reshape_input = False,reshape_output = (idx == len(self.block)-1),batch=batch)
        return l_o
