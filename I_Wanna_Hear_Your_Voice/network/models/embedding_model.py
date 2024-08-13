import torch
from resemblyzer import VoiceEncoder

class ResemblyzerVoiceEncoder:
    def __init__(self, device) -> None:
        self.model = VoiceEncoder(device)
        
    def __call__(self, audio: torch.Tensor):
        if audio.ndimension() == 1:
            return torch.tensor(self.model.embed_utterance(audio.numpy())).float().cpu()
        else:
            print(audio.shape)
            e = torch.stack([torch.tensor(self.model.embed_utterance(audio[i,:].numpy())).float().cpu() 
                             for i in range(audio.shape[0])])
            return e
