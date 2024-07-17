import torch.nn as nn
import torch


class LossWithPit:
    def __init__(self,n_speakers,loss_function):
        self.n_speakers = n_speakers
        self.loss_function = loss_function
    
    def __getAllPermutation(self):
        return 
    
class LossPit2(nn.Module):
    def __init__(self,loss_function):
        super().__init__()
        self.loss_function = loss_function
    def forward(self,yHat, label0,label1):
        yHat0 = yHat[:,0]
        yHat1 = yHat[:,1]
        l0 = self.loss_function(yHat0,label0) + self.loss_function(yHat1,label1)
        l1 = self.loss_function(yHat1,label0) + self.loss_function(yHat0,label1)
        return torch.min(l0,l1).mean()