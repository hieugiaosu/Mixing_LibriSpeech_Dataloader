import torch.nn as nn
from ..modules.tf_gridnet_modules import *
from ..modules.input_tranformation import STFTInput, RMSNormalizeInput
from ..modules.output_transformation import WaveGeneratorByISTFT, RMSDenormalizeOutput

class TF_Gridnet(nn.Module):
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
            ):
        super().__init__()

        self.input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )

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

        self.tf_gridnet_block = nn.Sequential(
            *[
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
    
    def forward(self,input):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        x = self.tf_gridnet_block(x)

        x = self.deconv(x)

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x
