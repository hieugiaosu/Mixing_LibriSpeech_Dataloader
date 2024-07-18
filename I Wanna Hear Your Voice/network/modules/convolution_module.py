import torch.nn as nn
from einops import rearrange

class SplitFeatureDeconv(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_srcs = 2,
            kernel_size_T = 3,
            kernel_size_F = 3,
            padding_T = 1,
            padding_F = 1,
            ) -> None:
        super().__init__()
        self.n_srcs = n_srcs
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * emb_dim, (kernel_size_F,kernel_size_T), padding=(padding_F,padding_T))
    def forward(self,input):
        output = self.deconv(input)
        output = rearrange(output,"B (N C) F T -> B C N F T", C=self.n_srcs)
        return output