import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
from network.modules.input_tranformation import STFTInput, RMSNormalizeInput
from network.modules.output_transformation import WaveGeneratorByISTFT, RMSDenormalizeOutput
from network.modules.tf_gridnet_modules import *
from network.modules.gate_module import BandFilterGate

class TFGridNetSE(nn.Module):
    def __init__(
            self,
            # n_srcs=2,
            n_fft=256,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            n_layers=6,
            input_kernel_size_T = 3,
            input_kernel_size_F = 3,
            output_kernel_size_T = 3,
            output_kernel_size_F = 3,
            lstm_hidden_units=256,
            attn_n_head=4,
            qk_output_channel=4,
            emb_dim=64,
            emb_ks=4,
            emb_hs=2,
            activation="PReLU",
            eps=1.0e-5,
            num_spks = 126
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
        self.gates = nn.ModuleList(
            [
                BandFilterGate(emb_dim,n_freqs)
                for _ in range(n_layers)
            ]
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

        


        self.aux_encoder = AuxEncoder(emb_dim, num_spks, n_freqs)

        self.deconv = TFGridnetDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.n_layers = n_layers
    def forward(self, mix, auxs):
        """
        Forward
        Args:
            mix (torch.Tensor): [B, -1]
            auxs (torch.Tensor): [B, -1]
        """
        audio_length = mix.shape[-1]
        aux_length = torch.tensor(auxs.shape[0])
        
        if mix.dim() == 2:
            x = mix.unsqueeze(1)

        if auxs.dim() == 2:
            a = auxs.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        a, std = self.input_normalize(a)

        a = self.stft(a)

        x = self.dimension_embedding(x) #[B, -1, F, T]
        a = self.dimension_embedding(a)
        
        a = a.transpose(2,3)
        n_freqs = x.shape[-2]
        
        a, speaker_pred = self.aux_encoder(a, aux_length)
        f = rearrange(a,"b (d q) -> b d q 1", q = n_freqs)
        b = rearrange(a,"b (d q) -> b d q 1", q = n_freqs)

        for i in range(self.n_layers):

            x = self.tf_gridnet_block[i](x) #[B, -1, F, T]
            x = self.gates[i](x,f,b)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x[:,0], speaker_pred
    

class AuxEncoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_spks,
                n_freqs):
        super(AuxEncoder, self).__init__()
        k1, k2 = (1, 3), (1, 3)
        self.d_feat = emb_dim
        self.n_freqs = n_freqs
        self.aux_enc = nn.ModuleList([EnUnetModule(emb_dim, emb_dim, (1, 5), k2, scale=4),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=3),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=2),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=1)])
        self.out_conv = nn.Linear(emb_dim, emb_dim * self.n_freqs)
        self.speaker = nn.Linear(emb_dim, num_spks)

    def forward(self,
                auxs: torch.Tensor,
                aux_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_lengths = (((aux_lengths // 3) // 3) // 3) // 3
        
        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [B, C, T, F]
        
        auxs = torch.stack([torch.mean(aux, dim = (1,2)) for aux in auxs], dim = 0)  # [B, C]
        
        auxs_out = self.out_conv(auxs)
        
        return auxs_out, self.speaker(auxs)


class FusionModule(nn.Module):
    def __init__(self,
                 emb_dim,
                 nhead=4,
                 dropout=0.1):
        super(FusionModule, self).__init__()
        self.nhead = nhead
        self.dropout = dropout
        param_size = [1, 1, emb_dim]

        self.attn = nn.MultiheadAttention(emb_dim,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        self.fusion = nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=1)
        self.alpha = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.zeros_(self.alpha)

    def forward(self,
                aux: torch.Tensor,
                esti: torch.Tensor) -> torch.Tensor:
        aux = aux.unsqueeze(1)  # [B, 1, C]
        flatten_esti = esti.flatten(start_dim=2).transpose(1, 2)  # [B, T*F, C]
        aux_adapt = self.attn(aux, flatten_esti, flatten_esti, need_weights=False)[0]
        aux = aux + self.alpha * aux_adapt  # [B, 1, C]

        aux = aux.unsqueeze(-1).transpose(1, 2).expand_as(esti)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]

        return esti


class EnUnetModule(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 scale: int):
        super(EnUnetModule, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.scale = scale

        self.in_conv = nn.Sequential(GateConv2d(cin, cout, k1, (1, 2)),
                                     nn.BatchNorm2d(cout),
                                     nn.PReLU(cout))
        self.encoder = nn.ModuleList([Conv2dUnit(k2, cout) for _ in range(scale)])
        self.decoder = nn.ModuleList([Deconv2dUnit(k2, cout, 1)])
        for i in range(1, scale):
            self.decoder.append(Deconv2dUnit(k2, cout, 2))
        self.out_pool = nn.AvgPool2d((3, 1))

    def forward(self, x: torch.Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)

        x = self.decoder[0](x)
        for i in range(1, len(self.decoder)):
            x = self.decoder[i](torch.cat([x, x_list[-(i + 1)]], dim=1))
        x_resi = x_resi + x

        return self.out_pool(x_resi)


class GateConv2d(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k: tuple,
                 s: tuple):
        super(GateConv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s

        self.conv = nn.Sequential(nn.ConstantPad2d((0, 0, k[0] - 1, 0), value=0.),
                                  nn.Conv2d(in_channels=cin,
                                            out_channels=cout * 2,
                                            kernel_size=k,
                                            stride=s))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)

        return outputs * gate.sigmoid()


class Conv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int):
        super(Conv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.conv = nn.Sequential(nn.Conv2d(c, c, k, (1, 2)),
                                  nn.BatchNorm2d(c),
                                  nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)


class Deconv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 expend_scale: int):
        super(Deconv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.expend_scale = expend_scale
        self.deconv = nn.Sequential(nn.ConvTranspose2d(c * expend_scale, c, k, (1, 2)),
                                    nn.BatchNorm2d(c),
                                    nn.PReLU(c))

    def forward(self, x):
        return self.deconv(x)