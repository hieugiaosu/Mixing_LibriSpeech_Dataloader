import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchaudio.transforms as T 
from einops import rearrange,repeat
import math
from resemblyzer import VoiceEncoder 

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

class Conv1dBlock(nn.Module):
    def __init__(self,inChannel,hiddenChannel,outChannel,windowSize,padding="same",stride=1,dilation=1, groups=1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv1d(inChannel,hiddenChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      ),
            nn.ELU(),
            nn.Conv1d(hiddenChannel,hiddenChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      ),
            nn.ELU(), 
            nn.Conv1d(hiddenChannel,outChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      )
        )
    def forward(self,x):
        return self.block(x)
class UnitBlock(nn.Module):
    def __init__(self,inChannel, outChannel,inputLength ,embHiddenChannel=171,embedingDim=512 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.audioTransform = Conv1dBlock(
            inChannel,
            (outChannel+inChannel)//2,
            outChannel,inputLength//100 + 1,
            )
        self.embTransform = Conv1dBlock(
            embHiddenChannel,
            (outChannel+embHiddenChannel)//2,
            outChannel,1
        )
        self.qExtract1 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,inputLength//100,stride=inputLength//100),
            nn.ELU(),
            nn.Linear(100,embedingDim)
        )
        self.vExtract1 = nn.Conv1d(outChannel,outChannel,1)
        self.crossAttention1 = EfficientAttention()
        self.layerNorm = nn.LayerNorm((outChannel,inputLength))
        self.convLayer = Conv1dBlock(outChannel,outChannel,outChannel,inputLength//100 + 1)
        self.qExtract2 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,inputLength//100,stride=inputLength//100),
            nn.ELU(),
            nn.Linear(100,embedingDim)
        )
        self.vExtract2 = nn.Conv1d(outChannel,outChannel,1)
        self.crossAttention2 = EfficientAttention()
    def forward(self,audio,emb_hidden):
        audio_i = self.audioTransform(audio)
        emb_i = self.embTransform(emb_hidden)
        q1 = self.qExtract1(audio_i)
        v1 = self.vExtract1(audio_i)
        att = self.crossAttention1(q=q1,k=emb_i,v=v1)
        o = self.convLayer(att)
        o = self.layerNorm(att+o)
        q2 = self.qExtract2(o)
        v2 = self.vExtract2(o)
        output = self.crossAttention2(q=q2,k=emb_i,v=v2)
        return output
class FiLMBlock(nn.Module):
    def __init__(self,inDim, outFiLMFeatures,transform ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(inDim,outFiLMFeatures*2),
            nn.ELU(),
            nn.Linear(outFiLMFeatures*2,outFiLMFeatures*2)
        )
        self.filmFeature = outFiLMFeatures
        self.transform = transform
    def forward(self,condition,*args,**kwwargs):
        film =self.model(condition)
        gamma = film[:,:self.filmFeature]
        beta = film[:,self.filmFeature:]
        y = self.transform(*args,**kwwargs)
        return gamma[:,:,None]*y+beta[:,:,None]
class AEBaseModel(nn.Module):
    def __init__(self,inputLength ,embHiddenChannel=171,embedingDim=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left = nn.ModuleList([
            UnitBlock(1,128,inputLength,embHiddenChannel,embedingDim),
            UnitBlock(128,256,inputLength//4,embHiddenChannel,embedingDim),
            UnitBlock(256,512,inputLength//16,embHiddenChannel,embedingDim)
        ])
        self.right = nn.ModuleList([
            UnitBlock(512,256,inputLength//16,embHiddenChannel,embedingDim),
            UnitBlock(256,128,inputLength//4,embHiddenChannel,embedingDim),
            UnitBlock(128,128,inputLength,embHiddenChannel,embedingDim)
        ])
        self.beforLast = FiLMBlock(embedingDim,128,nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.ELU()
        ))
        self.lastLayer = nn.Sequential(
            nn.Conv1d(128,128,inputLength//100+1,padding="same"),
            nn.ELU(),
            nn.Conv1d(128,1,inputLength//100+1,padding="same"),
            nn.Tanh()
        )
        self.downSample = nn.ModuleList([
            FiLMBlock(embedingDim,128,nn.AvgPool1d(4)),
            FiLMBlock(embedingDim,256,nn.AvgPool1d(4))
            ])
        self.middle = FiLMBlock(embedingDim,512,nn.Conv1d(512,512,1))
        self.upSample = nn.ModuleList([
            FiLMBlock(embedingDim,256,nn.ConvTranspose1d(256,256,4,stride=4)),
            FiLMBlock(embedingDim,128,nn.ConvTranspose1d(128,128,4,stride=4))
            ])
        self.norm = nn.ModuleList([
            nn.LayerNorm([256,inputLength//16]),
            nn.LayerNorm([256,inputLength//4]),
            nn.LayerNorm([128,inputLength]),
            ])
    def forward(self,audio,emb_hidden,emb):
        l1o = self.left[0](audio,emb_hidden)
        l2i = self.downSample[0](emb,l1o)
        l2o = self.left[1](l2i,emb_hidden)
        l3i = self.downSample[1](emb,l2o)
        l3o = self.left[2](l3i,emb_hidden)
        l4i = self.middle(emb,l3o)
        l4o = self.right[0](l4i,emb_hidden)
        l4o = self.norm[0](l4o+l3i)
        l5i = self.upSample[0](emb,l4o,output_size=l2o.size())
        l5i = self.norm[1](l5i+l2o)
        l5o = self.right[1](l5i,emb_hidden)
        l6i = self.upSample[1](emb,l5o,output_size=l1o.size())
        l6i = self.norm[2](l6i+l1o)
        l6o = self.right[2](l6i,emb_hidden)
        o = self.beforLast(emb,l6o)
        o = self.lastLayer(o) 
        return o
    
class AEInputConfigAfterEmbedding:
    def __init__(self) -> None:
        pass
    def __call__(self,e_output,audio):
        return {"audio": repeat(audio['mixing'],"b l -> (b r) 1 l", r=audio["audio"].shape[0]//audio["mixing"].shape[0]),
                "emb_hidden": e_output['last_hidden'], "emb": e_output['output']
                }


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

class SpeakerEmbedding:
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.encoder.eval()
    def __call__(self,wav ,*args, **kwds):
        return torch.stack(list(map(lambda x: torch.tensor(self.encoder.embed_utterance(x.numpy())),wav)))
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

    def forward(self,wav,emb):
        e = self.emb(emb)
        d1 = self.down1(wav,e)
        d2 = self.down2(d1,e)
        d3 = self.down3(d2,e)
        m1 = self.middle1(d3,e)
        m2 = self.middle2(m1,e)
        m3 = self.middle2(m2+m1,e)
        u1 = self.up1(m3+d3,e)
        u2 = self.up2(u1+d2,e)
        u3 = self.up3(u2+d1,e)
        y = self.outputLayer(u3)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else: 
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        batch = x.size(0)
        cache= torch.cat([self.pe]*batch,dim=0)
        x = torch.cat((x, cache[:,:seq_len,:]), dim=-1)
        return x

class TimeFrameCrossAttentionConvformerV2(nn.Module):
    def __init__(self, signalLength,signalChannel,embedingDim=512,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.e_ffn = nn.Sequential(
            nn.Linear(embedingDim,embedingDim),
            nn.LeakyReLU(),
            nn.Linear(embedingDim,embedingDim)
        )

        self.e_conv = nn.Sequential(
            nn.Conv1d(embedingDim,signalChannel,31,padding="same"),
            nn.SiLU(),
            nn.Conv1d(signalChannel,signalChannel,31,padding="same")
        )

        # self.e_skip = nn.Conv1d(1,signalChannel,1)

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
    
    def forward(self,x,e,origin_e):
        e1 = self.e_ffn(e)
        e2 = e+ e1/2
        # e2 = e2.unsqueeze(1)
        e3 = self.e_conv(e2)
        e4 = e3 + e2 #+ self.e_skip(e2)
        # e4 = e4.transpose(1,2)

        x1 = self.x_ffn(x)
        x2 = x + x1/2 
        x3 = self.x_conv(x2)
        x4 =x3+x2
        x4 = x4.transpose(1,2)

        att1 = self.cross_attention1(query=e4,key=x4,value=x4)[0]
        y1 = att1 + e4
        film = self.film(y1,origin_e)
        att2 = self.cross_attention2(query=e4,key=film,value=film)[0]
        y2 = att2 + e4
        y2 = y2.transpose(1,2)
        y3 = self.ffn(y2)
        y = self.norm(y3)
        return y
class UnetV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional = PositionalEncoding(511,512)
        
        self.emb = nn.Linear(256,512)
        self.down1 = DownConvBlock(1,128,32000,downWindow=2)
        self.down2 = DownConvBlock(128,128,16000,downWindow=2)
        self.down3 = DownConvBlock(128,256,8000)
        self.down4 = DownConvBlock(256,512,2000)
        self.middle1 = TimeFrameCrossAttentionConvformerV2(500,512,512)
        self.middle2 = TimeFrameCrossAttentionConvformerV2(500,512,512)
        self.middle3 = TimeFrameCrossAttentionConvformerV2(500,512,512)
        self.up1 = UpConvBlock(512,256,2000)
        self.up2 = UpConvBlock(256,128,8000)
        self.up3 = UpConvBlock(128,128,16000,upWindow=2)
        self.up4 = UpConvBlock(128,64,32000,upWindow=2)
        self.outputLayer = nn.Sequential(
            nn.Conv1d(64,1,1),
            nn.Tanh()
        )
        self.norm1 = nn.InstanceNorm1d(128)
        self.norm2 = nn.InstanceNorm1d(128)
        self.norm3 = nn.InstanceNorm1d(256)
        self.norm4 = nn.InstanceNorm1d(512)
    def forward(self,wav,emb):
        e = self.emb(emb)
        d1 = self.down1(wav,e)
        d2 = self.down2(d1,e)
        d3 = self.down3(d2,e)
        d4 = self.down4(d3,e)

        m_e = self.positional(e.unsqueeze(-1))

        m1 = self.middle1(d4,m_e,e)
        m2 = self.middle2(m1,m_e,e)
        m3 = self.middle2(m2+m1,m_e,e)

        u1_i = self.norm4(d4+m3)
        u1 = self.up1(u1_i,e)
        u2_i = self.norm3(u1+d3)
        u2 = self.up2(u2_i,e)
        u3_i = self.norm2(u2+d2)
        u3 = self.up3(u3_i,e)
        u4_i = self.norm1(u3+d1)
        u4 = self.up4(u4_i,e)
        y = self.outputLayer(u4)
        return y
