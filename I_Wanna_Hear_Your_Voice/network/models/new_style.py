import torch.nn as nn
from ..modules.tf_gridnet_modules import *
from ..modules.input_tranformation import STFTInput, RMSNormalizeInput
from ..modules.output_transformation import WaveGeneratorByISTFT, RMSDenormalizeOutput
from ..modules.gate_module import BandFilterGate
from ..modules.sequence_embed import SequenceEmbed, MutualAttention
class TFGridnetEncoder(nn.Module):
    def __init__(
            self,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            n_layers=6,
            input_kernel_size_T = 3,
            input_kernel_size_F = 3,
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

    def forward(self,input):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        x = self.tf_gridnet_block(x)

        return x, std, audio_length
    
class TFGridnetDecoder(nn.Module):
    def __init__(
            self,
            n_srcs=1,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_layers=6,
            output_kernel_size_T = 3,
            output_kernel_size_F = 3,
            lstm_hidden_units=192,
            attn_n_head=4,
            qk_output_channel=4,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="PReLU",
            eps=1.0e-5
            ):
        super().__init__()

        n_freqs = n_fft // 2 + 1

        self.istft = WaveGeneratorByISTFT(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window
        )

        self.output_denormalize = RMSDenormalizeOutput()

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


        self.gates = nn.ModuleList(
            [
                BandFilterGate(emb_dim,n_freqs)
                for _ in range(n_layers)
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
    def forward(self,x,f,b,std,audio_length):
        for i in range(len(self.tf_gridnet_block)):
            x = self.tf_gridnet_block[i](x)
            x = self.gates[i](x,f,b)
        
        x = self.deconv(x)
        x = rearrange(x,"B C N F T -> B N C F T")
        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)
        return x

class EncoderSPlitDecoder(nn.Module):
    def __init__(
            self,
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
            ) -> None:
        super().__init__()

        self.encoder = TFGridnetEncoder(
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_audio_channel=n_audio_channel,
            n_layers=n_layers//2,
            input_kernel_size_T=input_kernel_size_T,
            input_kernel_size_F=input_kernel_size_F,
            lstm_hidden_units=lstm_hidden_units,
            attn_n_head=attn_n_head,
            qk_output_channel=qk_output_channel,
            emb_dim=emb_dim,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            activation=activation,
            eps=eps
        )

        self.sequence_embed = SequenceEmbed(
            emb_dim=emb_dim,
            n_fft=n_fft,
            kernel_T=5,
            kernel_F=5
        )

        self.split = MutualAttention(
            emb_dim=emb_dim,
            n_head=attn_n_head,
            qk_output_channel=qk_output_channel,
            activation="PReLU",
            eps=0.00001
        )

        self.decoder = TFGridnetDecoder(
            n_srcs=1,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_layers=n_layers - n_layers//2,
            output_kernel_size_T=output_kernel_size_T,
            output_kernel_size_F=output_kernel_size_F,
            lstm_hidden_units=lstm_hidden_units,
            attn_n_head=attn_n_head,
            qk_output_channel=qk_output_channel,
            emb_dim=emb_dim,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            activation=activation,
            eps=eps
        )
    def forward(self,input,ref):
        print(input.shape,ref.shape)
        x, std, audio_length = self.encoder(input)
        ref_i ,_ ,_ = self.encoder(ref)

        print(x.shape,ref_i.shape)

        f,b = self.sequence_embed(x,ref_i)

        x = self.split(x,ref_i)

        x = self.decoder(x,f,b,std,audio_length)

        return x[:,0]