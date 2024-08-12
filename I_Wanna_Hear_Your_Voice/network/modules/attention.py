import torch
import torch.nn as nn
import torch.nn.functional as F
from .tf_gridnet_modules import AllHeadPReLULayerNormalization4DC, LayerNormalization
from einops import rearrange, repeat
import math


class IntraFrameCrossAttention(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_head = 4,
            qk_output_channel=12,
            activation="PReLU",
            eps = 1e-5
    ):
        super().__init__()
        assert emb_dim % n_head == 0
        E = qk_output_channel
        self.conv_Q = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_Q = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_K = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_K = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.norm_V = AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps)

        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim  
        self.n_head = n_head
    def forward(self,q,kv):
        """
        args:
            query (torch.Tensor): a query for cross attention, come frome the reference encoder
                                [B D Q Tq]
            kv (torch.Tensor): a key and value for cross attention, come frome the output of feature split
                                [B nSrc D Q Tkv]
        output:
            output: (torch.Tensor):[B D Q Tkv]
        """

        B, D, freq, Tq = q.shape

        _, nSrc, _, _, Tkv = kv.shape
        if Tq >= Tkv:
            q = q[:,:,:,-Tkv:]
        else: 
            r = math.ceil(Tkv/Tq)
            q = repeat(q,"B D Q T -> B D Q (T r)", r = r)
            q = q[:,:,:,-Tkv:]
        query = rearrange(q,"B D Q T -> B D T Q")
        kvInput = rearrange(kv,"B n D Q T -> B D T (n Q)")

        Q = self.norm_Q(self.conv_Q(query)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(kvInput)) # [B, n_head, C, T, Q*nSrc]
        V = self.norm_V(self.conv_V(kvInput)) 

        Q = rearrange(Q, "B H C T Q -> (B H T) Q C")
        K = rearrange(K, "B H C T Q -> (B H T) C Q").contiguous()
        _, n_head, channel, _, _ = V.shape
        V = rearrange(V, "B H C T Q -> (B H T) Q C")

        emb_dim = Q.shape[-1]
        qkT = torch.matmul(Q, K) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)

        att = torch.matmul(qkT,V)
        att = rearrange(att, "(B H T) Q C -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = B, T=Tkv)
        att = self.concat_proj(att)

        out = att + query
        out = rearrange(out, "B C T Q -> B C Q T")
        return out


class CrossFrameCrossAttention(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_head=4,
            qk_output_channel=4,
            activation="PReLU",
            eps = 1e-5

    ):
        super().__init__()
        assert emb_dim % n_head == 0
        E = qk_output_channel
        self.conv_Q = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_Q = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_K = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_K = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.norm_V = AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps)

        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim  
        self.n_head = n_head
    def forward(self,q,kv):
        """
        args:
            query (torch.Tensor): a query for cross attention, come frome the reference encoder
                                [B D Q Tq]
            kv (torch.Tensor): a key and value for cross attention, come frome the output of feature split
                                [B D Q Tkv]
        output:
            output: (torch.Tensor):[B D Q Tkv]
        """
        Tq = q.shape[-1]
        Tkv = kv.shape[-1]
        if Tq >= Tkv:
            q = q[:,:,:,-Tkv:]
        else: 
            r = math.ceil(Tkv/Tq)
            q = repeat(q,"B D Q T -> B D Q (T r)", r = r)
            q = q[:,:,:,-Tkv:]

        input = rearrange(q,"B C Q T -> B C T Q")
        kvInput = rearrange(kv,"B C Q T -> B C T Q")

        Q = self.norm_Q(self.conv_Q(input)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(kvInput))
        V = self.norm_V(self.conv_V(kvInput))
        
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
        out = att + input
        out = rearrange(out, "B C T Q -> B C Q T")
        return out

class CrossAttentionFilter(nn.Module):
    def __init__(self, emb_dim = 48) -> None:
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, q, k, v):
        """
        Args:
            q (torch.Tensor): from the provious layer, [B D F T]
            k (torch.Tensor): from the speaker embedidng encoder, [B D]
            v (torch.Tensor): from the speaker embedidng encoder, [B D]
        """

        B, D, _, T = q.shape

        q = rearrange(q, "B D F T -> (B T) F D")
        k = repeat(k, "B D -> (B T) D 1", T = T)
        v = repeat(v, "B D -> (B T) 1 D", T = T)

        qkT = torch.matmul(q, k)/(D**0.5)   # [(B T) F 1]
        qkT = F.softmax(qkT, dim=-1)
        att = torch.matmul(qkT, v)      # [(B T) F D]
        att = rearrange(att, "(B T) F D -> B D F T", B = B, T = T)
        return att
    
class CrossAttentionFilterV2(nn.Module):
    def __init__(self, emb_dim = 48) -> None:
        super().__init__()
        self.emb_dim = emb_dim
    def forward(self,q, kv):
        """
        Args:
        q: torch.Tensor, [B F D] a query for cross attention, come from the reference encoder (speaker embedding)
        kv: torch.Tensor, [B D F T] a key and value for cross attention, come from the output of previous layer (TF gridnet)
        """

        B, D, _, T = kv.shape

        Q = repeat(q, "B F D -> (B T) F D", T = T)
        K = rearrange(kv, "B D F T -> (B T) D F")
        V = rearrange(kv, "B D F T -> (B T) F D")

        qkT = torch.matmul(Q,K)/(D**0.5) #[(B T) F F]
        qkT = F.softmax(qkT, dim=-1)
        att = torch.matmul(qkT, V)      # [(B T) F D]
        att = rearrange(att, "(B T) F D -> B D F T", B = B, T = T)
        return att