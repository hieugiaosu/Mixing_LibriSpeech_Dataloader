import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .tf_gridnet_modules import CrossFrameSelfAttention

class SequenceEmbed(nn.Module):
    def __init__(
            self,
            emb_dim: int = 48,
            n_fft: int = 128,
            hidden_size: int = 192,
            kernel_T: int = 5,
            kernel_F: int = 5,
            ):
        super().__init__()

        self.n_freqs = n_fft // 2 + 1
        self.emb_dim = emb_dim

        self.conv = nn.Sequential(
            nn.Conv2d(emb_dim*2,emb_dim*2,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2),groups=emb_dim*2),
            nn.PReLU(),
            nn.Conv2d(emb_dim*2,emb_dim*2,1),
            nn.PReLU(),
            nn.Conv2d(emb_dim*2,emb_dim*2,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2),groups=emb_dim*2),
            nn.PReLU(),
            nn.Conv2d(emb_dim*2,emb_dim,1),
            nn.PReLU(),
        )

        self.linear_pre = nn.Conv1d(emb_dim*self.n_freqs,hidden_size,1)

        self.lstm = nn.LSTM(
            hidden_size,hidden_size,1,batch_first=True,bidirectional=True
        )

        self.linear = nn.Linear(hidden_size*2,emb_dim*self.n_freqs)

        self.filter_gen = nn.Conv1d(emb_dim,emb_dim,1)
        self.bias_gen = nn.Conv1d(emb_dim,emb_dim,1)
    def forward(self,x,ref):
        """
        Args:
            x: (B, D, F, T) input tensor from prevous layer
            ref: (B, D, F, T) embedding tensor previous layer
        """
        B, D, n_freq, T = x.shape
        input = torch.cat([x,ref],dim=1)
        input = self.conv(input)
        input = rearrange(input,'B D F T -> B (D F) T')
        input = self.linear_pre(input)
        input = rearrange(input,'B C T -> B T C')
        rnn , _ = self.lstm(input)
        feature = rnn[:,0]+rnn[:,-1] # (B, 2*Hidden)
        feature = self.linear(feature) # (B, D*F)
        feature = rearrange(feature,'B (D F) -> B D F',D=D,F=n_freq)
        f = self.filter_gen(feature)
        b = self.bias_gen(feature)

        return f.unsqueeze(-1), b.unsqueeze(-1)

class CrossFrameCrossAttention(CrossFrameSelfAttention):
    def __init__(self, emb_dim=48, n_freqs=65, n_head=4, qk_output_channel=4, activation="PReLU", eps=0.00001):
        super().__init__(emb_dim, n_freqs, n_head, qk_output_channel, activation, eps)
    
    def forward(self, q, kv):
        """
        Args:
            q: (B, D, F, T) query tensor
            kv: (B, D, F, T) key and value tensor
        """

        input_q = rearrange(q,"B C Q T -> B C T Q")
        input_kv = rearrange(kv,"B C Q T -> B C T Q")

        Q = self.norm_Q(self.conv_Q(input_q))
        K = self.norm_K(self.conv_K(input_kv))
        V = self.norm_V(self.conv_V(input_kv))
        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) (C Q) T").contiguous()
        batch, n_head, channel, frame, freq = V.shape
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")
        emb_dim = Q.shape[-1]
        qkT = torch.matmul(Q, K) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)
        att = torch.matmul(qkT,V)
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)
        out = att + input_q
        out = rearrange(out, "B C T Q -> B C Q T")
        return out
    
class MutualAttention(nn.Module):
    def __init__(self,kernel_T=5, kernel_F=5 ,emb_dim=48, n_freqs=65, n_head=4, qk_output_channel=4, activation="PReLU", eps=0.00001):
        super().__init__()

        self.ref_att = CrossFrameCrossAttention(emb_dim, n_freqs, n_head, qk_output_channel, activation, eps)
        self.tar_att = CrossFrameCrossAttention(emb_dim, n_freqs, n_head, qk_output_channel, activation, eps)

        self.mt_conv = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.PReLU(),
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.Sigmoid()
        )

        self.mr_conv = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.PReLU(),
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.Sigmoid()
        )

        self.mtr_conv = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.PReLU(),
            nn.Conv2d(emb_dim,emb_dim,(kernel_F,kernel_T),padding=(kernel_F//2,kernel_T//2)),
            nn.PReLU()
        )
    def forward(self,tar,ref):
        """
        Args:
            ref: (B, D, F, T) reference tensor
            tar: (B, D, F, T) target tensor
        """

        mr = self.ref_att(ref,tar)
        mt = self.tar_att(tar,ref)

        mrt = mr + mt

        mr = self.mr_conv(mr)
        mt = self.mt_conv(mt)
        mrt_o = self.mtr_conv(mrt)

        o = mr*mt*mrt_o + mrt
        return o