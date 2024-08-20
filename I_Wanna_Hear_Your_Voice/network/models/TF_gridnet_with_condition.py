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
        self.mix_input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )

        self.mix_dimension_embedding = DimensionEmbedding(
            audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size=(input_kernel_size_F,input_kernel_size_T),
            eps=eps
        )

        self.ref_dimension_embedding = DimensionEmbedding(
            audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size=(input_kernel_size_F,input_kernel_size_T),
            eps=eps
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
        
        mix_encode_layers = math.ceil(n_layers*2/3)
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

        self.split_layer = SplitFeatureDeconv(
            emb_dim=emb_dim,
            n_srcs=n_srcs,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
    
        self.intra_frame_cross_att = IntraFrameCrossAttention(
            emb_dim=emb_dim,
            n_head= attn_n_head,
            qk_output_channel=attn_n_head*3,
            activation=activation,
            eps=eps
            )
        self.cross_frame_cross_att = CrossFrameCrossAttention(
            emb_dim=emb_dim,
            n_head=attn_n_head,
            qk_output_channel=qk_output_channel,
            activation=activation,
            eps=eps
        )
        
        self.mix_decoder = nn.Sequential(
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
                ) for _ in range(mix_decode_layers)
            ]
        )

        self.deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.middle_deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=n_srcs,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
    
    def forward(self,mix,ref,middle = False):
        audio_length = mix.shape[-1]

        x = mix
        c = ref

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if c.dim() == 2:
            c = c.unsqueeze(1)
        x, std = self.mix_input_normalize(x)

        x = self.stft(x)

        c, _ = self.ref_input_normalize(c)

        c = self.stft(c)

        c = self.ref_dimension_embedding(c)

        c = self.ref_encoder(c)

        x = self.mix_dimension_embedding(x)

        x = self.mix_encoder(x)

        m = None
        if middle:
            m = self.middle_deconv(x)
            m = rearrange(m,"B C N F T -> B N C F T")
            m = self.istft(m,audio_length)
            m = self.output_denormalize(m,std)


        x = self.split_layer(x)

        x = self.intra_frame_cross_att(c,x)

        x = self.cross_frame_cross_att(c,x)

        x = self.mix_decoder(x)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)
        if middle: return x[:,0], m
        return x[:,0]
    
class DoubleChannelTFGridNet(TF_Gridnet):
    def __init__(self, 
                #  n_srcs=2, 
                 n_fft=128, 
                 hop_length=64, 
                 window="hann", 
                 n_audio_channel=1, 
                 n_layers=6, 
                 input_kernel_size_T=3, 
                 input_kernel_size_F=3, 
                 output_kernel_size_T=3, 
                 output_kernel_size_F=3, 
                 lstm_hidden_units=192, 
                 attn_n_head=4, 
                 qk_output_channel=4, 
                 emb_dim=48, 
                 emb_ks=4, 
                 emb_hs=1, 
                 activation="PReLU", 
                 eps=0.00001):
        super().__init__(1, n_fft, hop_length, window, n_audio_channel*2, n_layers, input_kernel_size_T, input_kernel_size_F, output_kernel_size_T, output_kernel_size_F, lstm_hidden_units, attn_n_head, qk_output_channel, emb_dim, emb_ks, emb_hs, activation, eps)
    def forward(self,input,condition):
        x = input
        c = condition

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if c.dim() == 2:
            c = c.unsqueeze(1)
        tc = c.shape[-1]
        tx = x.shape[-1]
        if tc >= tx:
            c = c[:,:,-tx:]
        else:
            n = math.ceil(tx/tc)
            c = repeat(c,"b c t -> b c (t n)",n=n)
            c = c[:,:,-tx:]
        
        mix_with_clue = torch.cat([x,c],dim=1)
        o = super().forward(mix_with_clue)
        return o[:,0]


class TargetSpeakerTF(nn.Module):
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

        self.film_layer = nn.ModuleList(
            [
                FiLMLayer(emb_dim,conditional_dim=conditional_dim,apply_dim=1)
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
        
        self.n_layers = n_layers
    
    def forward(self,input, clue):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)
        for i in range(self.n_layers):

            x = self.tf_gridnet_block[i](x)
            x = self.film_layer[i](x,clue)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x
    
class DoubleChannelTargetSpeakerTF(TargetSpeakerTF):
    def __init__(self, 
                #  n_srcs=2, 
                n_fft=128, 
                hop_length=64, 
                window="hann", 
                n_audio_channel=1, 
                n_layers=6, 
                input_kernel_size_T=3, 
                input_kernel_size_F=3, 
                output_kernel_size_T=3, 
                output_kernel_size_F=3, 
                lstm_hidden_units=192, 
                attn_n_head=4, 
                qk_output_channel=4, 
                emb_dim=48, 
                emb_ks=4, 
                emb_hs=1, 
                activation="PReLU", 
                eps=0.00001,
                conditional_dim = 256
                 ):
        super().__init__(1, n_fft, hop_length, window, n_audio_channel*2, n_layers, input_kernel_size_T, input_kernel_size_F, output_kernel_size_T, output_kernel_size_F, lstm_hidden_units, attn_n_head, qk_output_channel, emb_dim, emb_ks, emb_hs, activation, eps, conditional_dim)
    def forward(self, input, reference, embedding):
        x = input
        c = reference

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if c.dim() == 2:
            c = c.unsqueeze(1)
        tc = c.shape[-1]
        tx = x.shape[-1]
        if tc >= tx:
            c = c[:,:,-tx:]
        else:
            n = math.ceil(tx/tc)
            c = repeat(c,"b c t -> b c (t n)",n=n)
            c = c[:,:,-tx:]
        
        mix_with_clue = torch.cat([x,c],dim=1)
        o = super().forward(mix_with_clue,embedding)
        return o[:,0]
    
class FilterBandTFGridnet(nn.Module):
    def __init__(
            self,
            # n_srcs=2,
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
        n_freqs = n_fft//2 + 1
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

        self.filter_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)
        self.bias_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)


        self.gates = nn.ModuleList(
            [
                BandFilterGate(emb_dim,n_freqs)
                for _ in range(n_layers)
            ]
        )

        self.deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.n_layers = n_layers
    def forward(self,input, clue):
        audio_length = input.shape[-1]

        x = input
        print("input length")
        print(x.shape)
        print("input length 1")
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]
        f = self.filter_gen(clue)
        b = self.bias_gen(clue)
        f = rearrange(f,"b (d q) -> b d q 1", q = n_freqs)
        b = rearrange(b,"b (d q) -> b d q 1", q = n_freqs)

        x = x.transpose(-2, -1)
        for i in range(self.n_layers):

            x = self.tf_gridnet_block[i](x)
            x = self.gates[i](x,f,b)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x[:,0]
    
class FilterBandTFGridnetWithAttentionGate(nn.Module):
    def __init__(
            self,
            # n_srcs=2,
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
        n_freqs = n_fft//2 + 1
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

        self.query_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)


        self.attentions = nn.ModuleList(
            [
                CrossAttentionFilterV2(emb_dim)
                for _ in range(n_layers)
            ]
        )

        self.deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.n_layers = n_layers

    def forward(self,input, clue):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]

        q = self.query_gen(clue)
        q = rearrange(q,"b (d f) -> b f d", f=n_freqs)

        for i in range(self.n_layers):

            x = self.tf_gridnet_block[i](x)
            x = self.attentions[i](q,x)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x[:,0]