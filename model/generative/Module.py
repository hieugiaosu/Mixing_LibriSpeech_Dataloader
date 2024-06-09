import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchaudio.transforms as T 
from einops import rearrange,repeat


class Discriminator(nn.Module):
    def __init__(self):
        """
        we generate spectrogram with the parameter like this:
        n_fft = 512
        window_len = n_fft
        hop_length = 128
        => for the 2 seconds 16kHz audio the spectrogram will be at size:
        (512/2+1,[32000+128-1]/128+1) = (257,251)
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,5), #32,253,247
            nn.SiLU(),
            nn.Conv2d(32,64,5), #64,249, 243
            nn.SiLU(),
            nn.AvgPool2d(2) # 63, 124,121
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64,256,5),  #256, 120, 117
            nn.SiLU(),
            nn.Conv2d(256,512,5), #512, 116, 113
            nn.SiLU()
        )

        self.skipLink = nn.Conv2d(64,512,9) #512, 116, 113
        self.batchnorm = nn.BatchNorm2d(512)
        self.block3 = nn.Sequential(
            nn.AvgPool2d(2), #512,58,56
            nn.Conv2d(512,512,5), #512,54,52
            nn.SiLU(),
            nn.Conv2d(512,512,5), #512,50,48
            nn.SiLU(),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(4), #512,12,12
            nn.Dropout2d(0.4)
        )
        self.lin = nn.Sequential(
            nn.Linear(512*12*12,512),
            nn.SiLU(),
            nn.Linear(512,1)
        )
    def forward(self,x):
        batch = x.size(0)
        if x.dim() != 4:
            x = x.unsqueeze(1)
        o1 = self.block1(x)
        o2 = self.block2(o1)
        skip = self.skipLink(o1)
        o3 = self.batchnorm(skip+o2)
        o4 = self.block3(o3)
        o4 = o4.view(batch,-1)
        o = self.lin(o4)
        return o

class FeatureMapExtractor(nn.Module):
    def __init__(self,channel=2,height=257,width=251,processChannel = 128,dropout=0.2):
        super().__init__()
        self.processChannel = processChannel
        self.upChannel = nn.Sequential(
            nn.Conv2d(channel,processChannel//2,3,padding="same"),
            nn.SiLU(),
            nn.Conv2d(processChannel//2,processChannel,3,padding="same"),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.dil1 = nn.Sequential(
            nn.Conv2d(processChannel,processChannel,5,padding="same"),
            nn.SiLU()
        )

        self.dil3 = nn.Sequential(
            nn.Conv2d(processChannel,processChannel,5,dilation=(1,3),padding="same"),
            nn.SiLU()
        )

        self.dil5 = nn.Sequential(
            nn.Conv2d(processChannel,processChannel,5,dilation=(1,5),padding="same"),
            nn.SiLU()
        )

        self.dil7 = nn.Sequential(
            nn.Conv2d(processChannel,processChannel,5,dilation=(1,7),padding="same"),
            nn.SiLU()
        )

        self.residual = nn.Conv3d(1,4,(processChannel,5,5),padding=(0,2,2))
    def forward(self,x):
        i = self.upChannel(x)
        dil1 = self.dil1(i).unsqueeze(1)
        dil3 = self.dil3(i).unsqueeze(1)
        dil5 = self.dil5(i).unsqueeze(1)
        dil7 = self.dil7(i).unsqueeze(1)
        dil = torch.cat([dil1,dil3,dil5,dil7],dim = 1)
        res = self.residual(i.unsqueeze(1))
        o = dil*res
        o = o.mean(1)
        return i, F.normalize(o)

class BatchAdaptiveConv2d(nn.Module):
    def __init__(
            self,inChannels,outChannels,
            kernel=3,layer_pos_embedding_size = 256,
            condition_size=256
            ):
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weightsShape = [inChannels,outChannels]
        if isinstance(kernel,int):
            self.weightsShape = self.weightsShape+[kernel]*2
        else: 
            self.weightsShape.append(kernel[0])
            self.weightsShape.append(kernel[1])
        w = torch.empty(*self.weightsShape)
        nn.init.xavier_normal_(w)
        self.weights = nn.Parameter(w)
        b = torch.empty(outChannels,1)
        nn.init.xavier_normal_(b)
        self.bias = nn.Parameter(b.squeeze())

        self.weightAdapt = nn.Linear(layer_pos_embedding_size+condition_size,inChannels)
        self.biasAdapt = nn.Linear(layer_pos_embedding_size+condition_size,outChannels)
    def forward(self,x,condition,layer_pos_embedding):
        batch = x.size(0)
        i = torch.cat([condition,layer_pos_embedding],dim=-1)
        
        weightAdapt = self.weightAdapt(i) #shape (B,128)
        weightAdapt = weightAdapt[..., None, None, None]
        biasAdapt = self.biasAdapt(i) #shape (B,128)
        w = self.weights.expand(batch,-1,-1,-1,-1)
        w = w*weightAdapt #batch 128 128 3 3
        b = self.bias.expand(batch,self.outChannels)
        b = b*biasAdapt

        i = rearrange(x,'b c h w -> 1 (b c) h w')
        k = rearrange(w, 'g c_in c_out k1 k2 -> (g c_out) c_in k1 k2')

        o = F.conv2d(i,k,groups=batch,padding="same")
        o = rearrange(o, '1 (b c) h w -> b c h w', b=batch)
        o = o + b[...,None, None, None]
        return o



class GanModel(nn.Module):
    def __init__(
            self,channel=2,height=257,
            width=251,processChannel = 128,
            dropout=0.2,num_adaptive_conv_layers = 4,
            layers_position_embeding_dim=256,
            condition_dim = 256,
            adaptive_out_channels = [128,128,64,1]
        ):
        super().__init__()
        self.processChannel = processChannel
        self.num_adaptive_conv_layers = num_adaptive_conv_layers
        self.featureExtractor = FeatureMapExtractor(channel,height,width,processChannel,dropout)
        channels = [processChannel]+adaptive_out_channels
        self.globalInfo = nn.Sequential(
            nn.Conv2d(processChannel,condition_dim,(height,1)),
            nn.SiLU()
        )
        self.condAtt = nn.MultiheadAttention(condition_dim,2,batch_first=True)
        self.layerPosEmb = nn.Embedding(num_adaptive_conv_layers,layers_position_embeding_dim)
        
        self.adaptConv = nn.ModuleList([
            BatchAdaptiveConv2d(channels[i-1],channels[i],3,layers_position_embeding_dim,condition_dim) 
            for i in range(1,num_adaptive_conv_layers+1)
        ])
        
    def forward(self,spectrogram,condition):
        batch = spectrogram.size(0)
        device = spectrogram.device
        if spectrogram.dim()!=4:
            spectrogram = spectrogram.unsqueeze(1)
        i = torch.cat([spectrogram,torch.zeros_like(spectrogram)],dim=1)
        f1,f2 = self.featureExtractor(i)
        globalInfo = self.globalInfo(f1).squeeze()
        globalInfo = rearrange(globalInfo,"b d l -> b l d")

        cond = self.condAtt(condition.unsqueeze(1),globalInfo,globalInfo)[0]
        pos = torch.arange(0,self.num_adaptive_conv_layers,device=device)
        pos = self.layerPosEmb(pos)
        for idx,layer in enumerate(self.adaptConv):
            f2 = layer(f2,cond.squeeze(),pos[idx].expand(batch,-1))
        return f2