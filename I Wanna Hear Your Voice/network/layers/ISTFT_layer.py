import torch
import torch.nn as nn
from typing import Optional
from ..utils import ErrorMessageUtil
from einops import rearrange

class InverseSTFTLayer(nn.Module):
    def __init__(
            self,
            n_fft:int = 128,
            win_length: Optional[int] = None,
            hop_length:int = 64,
            window: str = "hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
            ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.window = getattr(torch,f"{window}_window")
    def forward(self,input,audio_length:int):
        """STFT forward function.
        Args:
            input: (Batch, Freq, Frames) or (Batch, Channels, Freq, Frames)
        Returns:
            output: (Batch, Nsamples) or (Batch, Channel, Nsample)
            
        Notice:
            input is a complex tensor
        """
        assert input.dim() == 4 or input.dim() == 3, ErrorMessageUtil.only_support_batch_input
        batch_size = input.size(0)
        multi_channel = (input.dim() == 4)
        if multi_channel:
            input = rearrange(input, "b c f t -> (b c) f t")
        window = self.window(
                    self.win_length,
                    dtype = input.real.dtype,
                    device = input.device
                )
        istft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                length = audio_length,
                return_complex = False
            )
        
        wave = torch.istft(input,**istft_kwargs)
        if multi_channel:
            wave = rearrange(wave,"(b c) l -> b c l", b = batch_size)
        return wave
    
class ComplexTensorLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(seal,input):
        assert input.shape[1] == 2, ErrorMessageUtil.complex_format_convert
        real = input[:,0]
        imag = input[:,1]

        return torch.complex(real,imag)