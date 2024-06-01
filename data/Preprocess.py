import torch 
from data.DataException import SilenceWindow

def SilenceCheck(level = 0.01,percentage = 0.5):
    def proc(audio:torch.Tensor):
        silence = (torch.abs(audio) < 0.01).float()
        if silence.mean() > percentage:
            raise SilenceWindow
        return silence
    return proc

def mix2Speakers(audio:torch.Tensor,silenceMap:torch.Tensor,label:torch.Tensor):
    invAudio = audio[-1::-1,:]
    invlabel = label[-1::-1]
    invSilenceMap = silenceMap[-1::-1,:]
    same = (invlabel == label).float()
    invSilenceMap = invSilenceMap*(1-same.unsqueeze(1))
    div = 2-silenceMap-invSilenceMap
    div = torch.max(div,torch.tensor(1.0))
    mix = (audio + invAudio*(1-same.unsqueeze(1)))/div 
    return mix