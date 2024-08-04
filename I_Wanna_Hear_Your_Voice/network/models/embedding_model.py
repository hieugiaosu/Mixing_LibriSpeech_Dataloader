import torch
from resemblyzer import VoiceEncoder

class ResemblyzerVoiceEncoder:
    def __init__(self,device) -> None:
        self.model = VoiceEncoder(device)
    def __call__(self, audio: torch.Tensor):
        if audio.size() == 1:
            return torch.tensor(self.model.embed_utterance(audio.numpy())).float().cpu()
        else:
            e = torch.stack([torch.tensor(self.model.embed_utterance(wav.numpy())).float().cpu() 
                            for wav in audio]
                            )
            return e