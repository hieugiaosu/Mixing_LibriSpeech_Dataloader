import torch
import torch.nn as nn
from typing import Optional
from ..utils import ErrorMessageUtil
from einops import rearrange

class STFTLayer(nn.Module):
    def __init__(
            self,
            n_fft:int = 128,
            win_length: Optional[int] = None,
            hop_length:int = 64,
            window: str = "hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
            pad_mode:str ="reflect"
            ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.pad_mode = pad_mode
        self.window = getattr(torch,f"{window}_window")
    def forward(self,input:torch.Tensor):
        """STFT forward function.
        Args:
            input: (Batch, Nsamples) or (Batch, Channel, Nsample)
        Returns:
            output: (Batch, Freq, Frames) or (Batch, Channels, Freq, Frames) 
        Notice:
            output is a complex tensor
        """
        assert input.dim() == 2 or input.dim() == 3, ErrorMessageUtil.only_support_batch_input
        batch_size = input.size(0)
        multi_channel = (input.dim() == 3)
        if multi_channel:
            input = rearrange(input, "b c l -> (b c) l")
        window = self.window(
                    self.win_length,
                    dtype = input.dtype,
                    device = input.device
                )

        stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode=self.pad_mode,
                return_complex=True
            )
        
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left
        stft_kwargs["window"] = torch.cat(
            [torch.zeros(n_pad_left,device=input.device), window, torch.zeros(n_pad_right,device=input.device)], 0
        )

        output = torch.stft(input,**stft_kwargs)
        if multi_channel:
            output = rearrange(output,"(b c) f t -> b c f t", b = batch_size)
        return output