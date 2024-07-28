import torch
import torch.nn as nn
from ..layers import InverseSTFTLayer, ComplexTensorLayer
from typing import Iterable, Optional

class WaveGeneratorByISTFT(nn.Module):
    def __init__(
            self,
            n_fft:int = 128,
            win_length: Optional[int] = None,
            hop_length:int = 64,
            window: str = "hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True
            ) -> None:
        super().__init__()
        self.istft = InverseSTFTLayer(
            n_fft,
            win_length,
            hop_length,
            window,
            center,
            normalized,
            onesided
        )

        self.float_to_complex = ComplexTensorLayer()

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,input,length:int=None):
        x = input
        if input.dtype in (torch.float16, torch.bfloat16):
            x = input.float()
        if x.dtype in (torch.float32,):
            x = self.float_to_complex(x)
        
        wav = self.istft(x,length)
        wav = wav.to(dtype=input.dtype)

        return wav

class RMSDenormalizeOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input,std):
        return input*std