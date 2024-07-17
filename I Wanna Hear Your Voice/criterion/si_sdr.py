import torch.nn as nn
import torch
import fast_bss_eval

class SI_SDR_Loss(nn.Module):
    def __init__(self,clamp_db=None,zero_mean=True,reduction="mean"):
        super().__init__()
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if reduction == "mean":
            self.loss_reduction = lambda loss: loss.mean(0)
        elif reduction == "sum":
            self.loss_reduction = lambda loss: loss.sum(0)
        else:
            self.loss_reduction = lambda loss: loss
    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        si_sdr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return self.loss_reduction(si_sdr)
    
class SI_SDR_Metric(SI_SDR_Loss):
    def __init__(self, clamp_db=None, zero_mean=True, reduction="mean"):
        super().__init__(clamp_db, zero_mean,reduction)
    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return -1*super().forward(ref, est)