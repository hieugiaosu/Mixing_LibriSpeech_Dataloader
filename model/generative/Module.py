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
        