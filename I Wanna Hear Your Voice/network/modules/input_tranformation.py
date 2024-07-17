import torch
import torch.nn as nn
from ..layers import STFTLayer
from ..utils import STFT_transform_type_enum
from typing import Iterable

class SimpleConv1DInput(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride = 1, 
                 padding = 0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', activation:str = 'ReLU'):
        super().__init__()
        activation = getattr(nn,activation)
        self.model = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel,stride,padding,dilation,groups,bias,padding_mode),
            activation()
        )
    def forward(self,input):
        return self.model(input)
    
class STFTInput(nn.Module):
    def __init__(
            self,
            n_fft: int = 128,
            win_length: int = None,
            hop_length: int = 64,
            window="hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
            spec_transform_type: str = None,
            spec_factor: float = 0.15,
            spec_abs_exponent: float = 0.5,
        ):
        super().__init__()
        self.stft = STFTLayer(
            n_fft,
            win_length,
            hop_length,
            window,
            center,
            normalized,
            onesided
            )
        
        self.spec_transform_type = spec_transform_type
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
    
        self.spec_transform = lambda spec: spec
        if self.spec_transform_type == STFT_transform_type_enum.exponent:
            self.spec_transform = lambda spec: spec.abs() ** self.spec_abs_exponent * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == STFT_transform_type_enum.log:
            self.spec_transform = lambda spec: torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle()) * self.spec_factor
        
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,input):
        """
        Notice that, in pytorch, the STFT does not support quantize 16 bit float, so this function
        is decorated with @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        Args:
            input (torch.Tensor): signal [Batch, Nsamples] or [Batch,channel,Nsamples]
        ouputs:
            spectrum (torch.Tensor): float tensor perform the spectrum with 2 channel, the first channel
                is real part of spectrum, the second channel is the imaginary part of spectrum
                [Batch, 2, F, T] or [Batch, 2 * channel, F, T]
        """
        
        spectrum = self.stft(input.float())
        spectrum = self.spec_transform(spectrum)

        re = spectrum.real
        im = spectrum.imag

        if input.dim() == 2:
            re = re.unsqueeze(1)
            im = im.unsqueeze(1)

        if input.dtype in (torch.float16, torch.bfloat16):
            re = re.to(dtype=input.dtype)
            im = im.to(dtype=input.dtype)

        return torch.cat([re,im],dim=1)

class RMSNormalizeInput(nn.Module):
    def __init__(self, dim: Iterable[int], keepdim:bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self,input):
        std = torch.std(input,dim=self.dim,keepdim=self.keepdim)
        output = input/std
        return output, std