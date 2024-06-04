import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchaudio.transforms as T 
from einops import rearrange,repeat
from functools import reduce
import numpy as np
class ModuleWithPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def get_sinusoidal_positional_encoding(self, max_len, d_model,device=None):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pos_encoding = torch.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.to(device)
        return pos_encoding
class SpeakerEmbeddingInputTransform(nn.Module):
    def __init__(self,melSpectrogramParam:dict ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mel = T.MelSpectrogram(**melSpectrogramParam)
    def forward(self,audio):
        mel = self.mel(audio)
        mel = torch.log(1+mel)
        return F.tanh(mel)

class SpectrogramTransformer(ModuleWithPositionalEncoding):
    def __init__(self,t=500,f=64,f_window_size=32,t_window_size=125) -> None:
        super().__init__()
        assert t%t_window_size == 0 and f%f_window_size == 0, "expect t%t_window_size == 0 and f%f_window_size == 0"
        self.f_window_size = f_window_size
        self.t_window_size = t_window_size
        self.t = t 
        self.f = f
        self.fconv = nn.Conv2d(1,t_window_size, (1,t_window_size),stride=(1,t_window_size))
        self.fq = nn.Linear(t_window_size*2,t_window_size*2)
        self.fk = nn.Linear(t_window_size*2,t_window_size*2)
        self.fv = nn.Linear(t_window_size*2,t_window_size*2)
        self.fattention = nn.MultiheadAttention(t_window_size*2,5,batch_first=True)
        self.fattnl = nn.Linear(t_window_size*2,t_window_size)

        self.tconv = nn.Conv2d(1,f_window_size,(f_window_size,1),stride=(f_window_size,1))
        self.tq = nn.Linear(f_window_size*2,f_window_size*2)
        self.tk = nn.Linear(f_window_size*2,f_window_size*2)
        self.tv = nn.Linear(f_window_size*2,f_window_size*2)
        self.tattention = nn.MultiheadAttention(f_window_size*2,4,batch_first=True)
        self.tattnl = nn.Linear(f_window_size*2,f_window_size)
    def forward(self,x):
        batch_size = x.size(0)
        assert x.size(1) == self.f, f"expect f={self.t} but get {x.size(1)}"
        assert x.size(2) == self.t, f"expect t={self.t} but get {x.size(2)}"
        ## x: (B,F,T) example (B,64,500)
        i = x.unsqueeze(1)
        i = self.fconv(i)
        i = rearrange(i,"b d f l -> (b l) f d")
        fpos = self.get_sinusoidal_positional_encoding(
                i.size(1),
                i.size(2) if i.size(2)%2==0 else i.size(2)+1,
                i.device
                ).expand(i.size(0),-1,-1)
        if i.size(2)%2!=0:
            fpos = fpos[:,:,:-1] 
        fi = torch.cat([i,fpos],dim=-1)
        fq = self.fq(fi)
        fk = self.fk(fi)
        fv = self.fv(fi)
        fatt = self.fattention(fq,fk,fv)[0]
        fo = F.silu(self.fattnl(F.normalize(fatt+fi)))
        fo = rearrange(fo,"(b l) f d -> b 1 f (l d)",b=batch_size)

        i2 = self.tconv(fo)
        i2 = rearrange(i2,"b d l t -> (b l) t d")
        tpos = self.get_sinusoidal_positional_encoding(
                i2.size(1),
                i2.size(2) if i2.size(2)%2==0 else i2.size(2)+1,
                i2.device
                ).expand(i2.size(0),-1,-1)
        if i2.size(2)%2!=0:
            tpos = tpos[:,:,:-1]
        ti = torch.cat([i2,tpos],dim=-1) 
        tq = self.tq(ti)
        tk = self.tk(ti)
        tv = self.tv(ti)
        tatt = self.tattention(tq,tk,tv)[0]
        to = F.silu(self.tattnl(F.normalize(tatt+ti)))
        to = rearrange(to, "(b l) t d -> b (d l) t",b=batch_size)
        return to
    

class SpeakerEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputTransform = SpeakerEmbeddingInputTransform({
            "sample_rate":16000,
            "n_fft": 512,
            "f_min":40,
            "f_max":4000,
            "n_mels": 128,
            "hop_length":160
        })
        self.conv = nn.Conv1d(128,128,2)
        self.transformer = nn.Sequential(
            SpectrogramTransformer(1000,128,32,125),
            SpectrogramTransformer(1000,128,64,250),
            SpectrogramTransformer(1000,128,128,500),
            SpectrogramTransformer(1000,128,128,1000)
        )
        self.linear = nn.Linear(128,256)
    def forward(self,audio):
        mel = self.inputTransform(audio)
        mel = mel.squeeze()
        mel = F.silu(self.conv(mel))
        # print(mel.shape)
        e = self.transformer(mel)
        e = rearrange(e,"b f t -> b t f")[:,-1,:]
        e = self.linear(e)
        return e
class SI_SDRLoss(nn.Module):
    '''num batch dim is how many dim in tensor is a batch. 
    for example input can be (B,L) with batch is batch size and L is audio Length
    or it can be: (B,S,L) with batch is batch size, S is number of speaker in each
    batch and L is audio Length.
    For this version, I only support those kind of shape. So numBatchDim is only
    2 or 3
    '''
    def __init__(self,numBatchDim:int=2 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert numBatchDim == 2 or numBatchDim == 3, "only support 2 or 3"
        self.numBatchDim = numBatchDim
    def forward(self,input,label):
        assert input.shape == label.shape, "2 input of loss function must have the same shape"
        assert input.dim() == self.numBatchDim, f"numBatchDim is set to {self.numBatchDim} but get {input.shape} tensor as a input"
        if self.numBatchDim == 3:
            ## now is (B,S,L)
            input = rearrange(input,"b s l -> (b s) l")
            label = rearrange(label,"b s l -> (b s) l")
        term1 = torch.bmm(input.unsqueeze(1),label.unsqueeze(2)).squeeze()
        term2 = torch.bmm(label.unsqueeze(1),label.unsqueeze(2)).squeeze()
        alpha = term1/(term2+1e-6) 
        term3 = alpha.unsqueeze(1)*label-input
        term4 = torch.bmm(term3.unsqueeze(1),term3.unsqueeze(2)).squeeze() + 1e-6
        loss = -10*torch.log10(((alpha**2)*term2 + 1e-6)/term4)
        return loss.mean(0)

class EfficientAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)    
    def forward(self,q,k,v):
        q_n = F.softmax(q,dim=-1)
        k_n = F.softmax(k,dim=-1)
        k_n = k_n.transpose(1,2)
        return torch.bmm(q_n,torch.bmm(k_n,v))

class FiLMLayer(nn.Module):
    def __init__(self,featureSize,transformSize ,applyDim,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.applyDim = applyDim
        self.alpha_extract = nn.Sequential(
            nn.Linear(featureSize,featureSize),
            nn.LeakyReLU(),
            nn.Linear(featureSize,transformSize)
        )
        self.beta_extract = nn.Sequential(
            nn.Linear(featureSize,featureSize),
            nn.LeakyReLU(),
            nn.Linear(featureSize,transformSize)
        )
    def forward(self,x,feature):
        alpha = self.alpha_extract(feature)
        beta = self.beta_extract(feature)
        if self.applyDim != 1: 
            x = x.transpose(1,self.applyDim)
        while alpha.dim() != x.dim():
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        y = alpha*x + beta 
        if self.applyDim != 1: 
            y = y.transpose(1,self.applyDim)
        return y 

class DownConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,signalLength,embedingDim=512,downWindow=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv1d(inChannel,(outChannel+inChannel)//2,33,padding="same"),
            nn.ELU(),
            nn.Conv1d((outChannel+inChannel)//2,(outChannel+inChannel)//2,33,padding="same"),
            nn.ELU(),
            nn.Conv1d((outChannel+inChannel)//2,outChannel,33,padding="same"),
            nn.InstanceNorm1d(outChannel,affine=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU(),
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU()
        )

        self.skipLinkNorm = nn.InstanceNorm1d(outChannel,affine=True)
        self.film = FiLMLayer(embedingDim,outChannel,1)

        self.conv3 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU(),
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU()
        )
        self.skipLinkLayerNorm = nn.LayerNorm([outChannel,signalLength])
        self.downSampling = nn.AvgPool1d(downWindow)
        self.filmDown = FiLMLayer(embedingDim,outChannel,1)
    def forward(self,x,e):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.skipLinkLayerNorm(y1+y2)
        y4 = self.film(y3,e)
        y5 = self.conv3(y4)
        y6 = self.skipLinkLayerNorm(y5+y4)
        y7 = self.downSampling(y6)
        y = self.filmDown(y7,e)
        return y
class UpConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,signalLength,embedingDim=512,upWindow=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upSampling = nn.ConvTranspose1d(inChannel,inChannel,upWindow,stride=upWindow)
        self.filmUp = FiLMLayer(embedingDim,inChannel,1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(inChannel,(outChannel+inChannel)//2,33,padding="same"),
            nn.ELU(),
            nn.Conv1d((outChannel+inChannel)//2,(outChannel+inChannel)//2,33,padding="same"),
            nn.ELU(),
            nn.Conv1d((outChannel+inChannel)//2,outChannel,33,padding="same"),
            nn.InstanceNorm1d(outChannel,affine=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU(),
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU()
        )

        self.skipLinkNorm = nn.InstanceNorm1d(outChannel,affine=True)
        self.film = FiLMLayer(embedingDim,outChannel,1)

        self.conv3 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU(),
            nn.Conv1d(outChannel,outChannel,33,padding="same"),
            nn.SiLU()
        )
        self.skipLinkLayerNorm = nn.LayerNorm([outChannel,signalLength])
        
    def forward(self,x,e):
        x1 = self.upSampling(x)
        x2 = self.filmUp(x1,e)
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y3 = self.skipLinkLayerNorm(y1+y2)
        y4 = self.film(y3,e)
        y5 = self.conv3(y4)
        y = self.skipLinkLayerNorm(y5+y4)
        return y
class TimeFrameCrossAttentionConvformer(nn.Module):
    def __init__(self, signalLength,signalChannel,embedingDim=512,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_ffn = nn.Sequential(
            nn.Linear(embedingDim,embedingDim),
            nn.LeakyReLU(),
            nn.Linear(embedingDim,embedingDim)
        )

        self.e_conv = nn.Sequential(
            nn.Conv1d(1,signalChannel,31,padding="same"),
            nn.SiLU(),
            nn.Conv1d(signalChannel,signalChannel,31,padding="same")
        )

        self.e_skip = nn.Conv1d(1,signalChannel,1)

        self.x_ffn = nn.Sequential(
            nn.Linear(signalLength,signalLength),
            nn.LeakyReLU(),
            nn.Linear(signalLength,signalLength)
        )

        self.x_conv = nn.Sequential(
            nn.Conv1d(signalChannel,signalChannel,31,padding="same"),
            nn.SiLU(),
            nn.Conv1d(signalChannel,signalChannel,31,padding="same")
        )

        self.cross_attention1 = nn.MultiheadAttention(signalChannel,4,batch_first=True)
        self.film = FiLMLayer(embedingDim,embedingDim,1)
        self.cross_attention2 = nn.MultiheadAttention(signalChannel,4,batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(signalChannel,signalLength),
            nn.LeakyReLU(),
            nn.Linear(signalLength,signalLength),
            nn.Tanh()
        )

        self.norm = nn.LayerNorm([signalChannel,signalLength])
    
    def forward(self,x,e):
        e1 = self.e_ffn(e)
        e2 = e+ e1/2
        e2 = e2.unsqueeze(1)
        e3 = self.e_conv(e2)
        e4 = e3 + self.e_skip(e2)
        e4 = e4.transpose(1,2)

        x1 = self.x_ffn(x)
        x2 = x + x1/2 
        x3 = self.x_conv(x2)
        x4 =x3+x2
        x4 = x4.transpose(1,2)

        att1 = self.cross_attention1(query=e4,key=x4,value=x4)[0]
        y1 = att1 + e4
        film = self.film(y1,e)
        att2 = self.cross_attention2(query=e4,key=film,value=film)[0]
        y2 = att2 + e4
        y2 = y2.transpose(1,2)
        y3 = self.ffn(y2)
        y = self.norm(y3)
        return y

class Unet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Linear(256,512)
        self.down1 = DownConvBlock(1,128,32000)
        self.down2 = DownConvBlock(128,256,8000)
        self.down3 = DownConvBlock(256,512,2000)
        self.middle1 = TimeFrameCrossAttentionConvformer(500,512,512)
        self.middle2 = TimeFrameCrossAttentionConvformer(500,512,512)
        self.middle3 = TimeFrameCrossAttentionConvformer(500,512,512)
        self.up1 = UpConvBlock(512,256,2000)
        self.up2 = UpConvBlock(256,128,8000)
        self.up3 = UpConvBlock(128,64,32000)
        self.outputLayer = nn.Sequential(
            nn.Conv1d(64,1,1),
            nn.Tanh()
        )

    def forward(self,wav,emb,teacher=False,teacher_wav=None,return_latent=False):
        if not teacher:
            e = self.emb(emb)
            d1 = self.down1(wav,e)
            d2 = self.down2(d1,e)
            d3 = self.down3(d2,e)
            m1 = self.middle1(d3,e)
            m2 = self.middle2(m1,e)
            m3 = self.middle3(m2+m1,e)
            u1 = self.up1(m3+d3,e)
            u2 = self.up2(u1+d2,e)
            u3 = self.up3(u2+d1,e)
            y = self.outputLayer(u3)
            if return_latent: return y,m3
            return y
        else: 
            assert wav.shape == teacher_wav.shape, "mismatch shape"
            batch = wav.shape[0]
            e = torch.cat([emb]*2,dim=0)
            i = torch.cat([teacher_wav,wav],dim=0)
            e = self.emb(e)
            d1 = self.down1(i,e)
            d2 = self.down2(d1,e)
            d3 = self.down3(d2,e)
            m1 = self.middle1(d3,e)
            m2 = self.middle2(m1,e)
            m3 = self.middle3(m2+m1,e)

            teacher_latent = torch.cat([m3[:batch,:,:]]*2,dim=0)
            u1 = self.up1(teacher_latent,e)
            u2 = self.up2(u1+d2,e)
            u3 = self.up3(u2+d1,e)
            y = self.outputLayer(u3)
            if return_latent: return y,m3
            return y

class ContinuousEmbeddingLayer(nn.Module):
    def __init__(self,embedding_dim,chunks=1000):
        super().__init__()
        self.chunks = float(chunks)
        self.emb = nn.Embedding(chunks,embedding_dim)
    def forward(self, x):
        idx = (F.tanh(x)+1)*self.chunks/2
        idx = idx.long()
        e = self.emb(idx)
        return e

class TimeFrameFeatureExtractingBlockAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.att = nn.MultiheadAttention(dim,2,batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding="same"),
            nn.SiLU(),
            nn.Conv2d(32,64,3,padding="same"),
            nn.SiLU(),
            nn.Conv2d(64,1,3,padding="same"),
        )
    def forward(self,x,e):
        ### now x have shape (B,T,F)
        att = self.att(x,e,e)[0]
        convi = F.normalize(att+x)
        convi = convi.unsqueeze(1)
        o = self.conv(convi)
        o = o.squeeze()
        return o



class TimeFrameFeatureExtractingBlock(ModuleWithPositionalEncoding):
    def __init__(self,dim,nLayer):
        super().__init__()
        self.dim = dim
        self.nLayer = nLayer
        self.embLayer = ContinuousEmbeddingLayer(dim*2)

        self.q_transform = nn.Sequential(
            nn.Linear(dim*2,dim*2),
            nn.SiLU()
        )

        self.transformer = nn.ModuleList([
            TimeFrameFeatureExtractingBlockAttention(dim*2) for _ in range(nLayer)
        ])

        self.linear = nn.Linear(dim*2,dim)

    def forward(self,spectrogram,embedding):
        #spectrogram shape (B,F,T)
        i = rearrange(spectrogram,"B F T -> B T F")
        e = self.embLayer(embedding)
        pos = self.get_sinusoidal_positional_encoding(
                i.size(1),
                i.size(2) if i.size(2)%2==0 else i.size(2)+1,
                i.device
                ).expand(i.size(0),-1,-1)
        if i.size(2)%2!=0:
            pos = pos[:,:,:-1] 
        i = torch.cat([i,pos],dim=-1)
        o = self.q_transform(i)
        for layer in self.transformer:
            o = layer(o,e)
        o = self.linear(o)
        o = rearrange(o, "B T F -> B F T")
        return o

class DownUnetConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,embeddingDim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannel,outChannel,3),
            nn.SiLU(),
            nn.Conv2d(outChannel,outChannel,3),
            nn.SiLU(),
            nn.Conv2d(outChannel,outChannel,3),
            nn.SiLU()
        )
        self.down = nn.MaxPool2d(2)
        self.filmDown = FiLMLayer(embeddingDim,outChannel,1)
    def forward(self,x,e):
        o = self.conv(x)
        o = self.down(o)
        o = self.filmDown(o,e)
        return o
    
class UpUnetConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,embeddingDim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel,3),
            nn.SiLU(),
            nn.ConvTranspose2d(outChannel,outChannel,3),
            nn.SiLU(),
            nn.ConvTranspose2d(outChannel,outChannel,3),
            nn.SiLU()
        )
        self.up = nn.ConvTranspose2d(outChannel,outChannel,2,stride=2)
        self.filmUp = FiLMLayer(embeddingDim,outChannel,1)
    def forward(self,x,e):
        o = self.conv(x)
        o = self.up(o)
        o = self.filmUp(o,e)
        return o
    
class MiddleUnetConvBlock(nn.Module):
    def __init__(self,inChannel,outChannel,embeddingDim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannel,outChannel,3,padding="same"),
            nn.SiLU(),
            nn.Conv2d(outChannel,outChannel,3,padding="same"),
            nn.SiLU(),
            nn.Conv2d(outChannel,outChannel,3,padding="same"),
            nn.SiLU()
        )
        self.filmUp = FiLMLayer(embeddingDim,outChannel,1)
    def forward(self,x,e):
        o = self.conv(x)
        o = self.filmUp(o,e)
        return o
    
class UnetConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownUnetConvBlock(1,64)
        self.down2 = DownUnetConvBlock(64,128)
        self.middle = MiddleUnetConvBlock(128,512)
        self.up1 = UpUnetConvBlock(512,32)
        self.up2 = UpUnetConvBlock(32,1)
    def forward(self,x,e):
        i = x.unsqueeze(1)
        o = self.down1(i,e)
        o = self.down2(o,e)
        o = self.middle(o,e)
        o = self.up1(o,e)
        o = self.up2(o,e)
        return o.squeeze()
    
class SpeechSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = SpeakerEmbedding()
        self.timeExtract = TimeFrameFeatureExtractingBlock(257,3)
        self.unet = UnetConv2d()
    def forward(self,audio_sample, mixed_spectrogram):
        e = self.emb(audio_sample)
        o = self.timeExtract(mixed_spectrogram,e)
        o = self.unet(o,e)
        return o,e