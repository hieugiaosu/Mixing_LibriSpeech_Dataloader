import torch.nn as nn
from typing import Tuple
class DimensionEmbedding(nn.Module):
    def __init__(
            self, audio_channel:int = 1,emb_dim:int = 48,
            kernel_size: Tuple[int,int] = (3,3),
            padding = "same",eps=1.0e-5
            ) -> None:
        super().__init__()
        self.emb = nn.Sequential(
            nn.Conv2d(audio_channel, emb_dim, kernel_size,padding=padding),
            nn.GroupNorm(1,emb_dim,eps=eps)
        )
    def forward(self,input):
        return self.emb(input)