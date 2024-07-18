import torch.nn as nn
from ..modules.tf_gridnet_modules import *
from ..modules.input_tranformation import STFTInput, RMSNormalizeInput
from ..modules.output_transformation import WaveGeneratorByISTFT, RMSDenormalizeOutput

class TFGridFormer(nn.Module):
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
            eps=1.0e-5,):
        super().__init__()
        self.ref_input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.ref_stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )

        self.mix_input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.mix_stft = STFTInput(
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

        self.ref_encoder = TFGridnetBlock(
                    emb_dim=emb_dim,
                    kernel_size=emb_ks,
                    emb_hop_size=emb_hs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=qk_output_channel,
                    activation=activation,
                    eps=eps
                )
        
        mix_encode_layers = n_layers//2
        mix_decode_layers = n_layers - mix_encode_layers

        self.mix_encoder = nn.Sequential(
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
                ) for _ in range(mix_encode_layers)
            ]
        )

        self.split_layer = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=n_srcs,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )