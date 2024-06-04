import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from data.DataException import SilenceWindow

# def SilenceCheck(level = 0.01,percentage = 0.5):
#     def proc(audio:torch.Tensor):
#         silence = (torch.abs(audio) < 0.01).float()
#         if silence.mean() > percentage:
#             raise SilenceWindow
#         return silence
#     return proc
def Mix2Speakers():
    spec = T.Spectrogram(512,hop_length=160)
    def mixAudio(audio:torch.Tensor,paddingMap:torch.Tensor,label:torch.Tensor):
        invAudio = torch.flip(audio,[0])
        invlabel = torch.flip(label,[0])
        audioAct = 1 - paddingMap
        invAudioAct = torch.flip(audioAct,[0])
        diff = (invlabel != label).float()
        invAudioAct = invAudioAct*diff.unsqueeze(1)
        sumTerm = audioAct+invAudioAct 
        sumTerm = torch.max(sumTerm,torch.tensor(1.0))
        mix = audio*(audioAct/sumTerm) + invAudio*(invAudioAct/sumTerm)
        return mix, F.normalize(spec(mix))
    return mixAudio