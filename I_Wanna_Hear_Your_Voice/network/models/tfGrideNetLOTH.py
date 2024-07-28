import torch.nn as nn
from ..modules.tf_gridnet_modules import *
from ..modules.input_tranformation import STFTInput, RMSNormalizeInput
from ..modules.output_transformation import WaveGeneratorByISTFT, RMSDenormalizeOutput
from ..modules.convolution_module import SplitFeatureDeconv
from ..modules.attention import *
from .TF_gridnet import TF_Gridnet
from ..layers import FiLMLayer
from ..modules.gate_module import BandFilterGate
from einops import rearrange, repeat
import math
class TargetSpeakerLOTH(nn.Module):
    def __init__(
            self,
            n_srcs=2,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            n_layers=6,
            input_kernel_size_T = 3,
            input_kernel_size_F = 3,
            output_kernel_size_T = 3,
            output_kernel_size_F = 3,
            lstm_hidden_units=192,
            attn_n_head=4,
            qk_output_channel=4,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="PReLU",
            eps=1.0e-5,
            conditional_dim = 256
            ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )
        n_freqs = n_fft//2 + 1
        
        self.istft = WaveGeneratorByISTFT(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window
        )

        self.output_denormalize = RMSDenormalizeOutput()

        self.dimension_embedding = DimensionEmbedding(
            audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size=(input_kernel_size_F,input_kernel_size_T),
            eps=eps
        )

        self.tf_gridnet_block = nn.ModuleList(
            [
                TFGridnetBlock(
                    emb_dim=emb_dim,
                    kernel_size=emb_ks,
                    emb_hop_size=emb_hs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=qk_output_channel,
                    activation=activation,
                    eps=eps
                ) for _ in range(n_layers)
            ]
        )

        self.deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=n_srcs,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.n_layers = n_layers

        self.embed_to_feats_proj = nn.Sequential(
            nn.Linear(conditional_dim, emb_dim * n_freqs),
            nn.LayerNorm(emb_dim * n_freqs)
        )
    
    def forward(self,input, spk_emb):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x) #[B, -1, T, F]

        embed = self.embed_to_feat_proj(spk_emb) #[B,C*F]
        embed = embed.reshape([x.shape(0), self.emb_dim, x.shape(3)]).unsqueeze(2) #[B, C, 1, F]

        for i in range(self.n_layers):
            if i==1:
                x = x*embed
            x = self.tf_gridnet_block[i](x)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x