import torch
import torchaudio
import numpy as np
import pandas as pd
from .cluster import Cluster
from einops import repeat

class WhamNoiseAugmentation:
    def __init__(
            self,
            noise_dir: str,
            noise_meta: str,
            noise_snr_range: tuple,
            sampling_rate: int,
            noise_duration: float,
            ) -> None:
        self.noise_dir = noise_dir
        self.noise_meta = pd.read_csv(noise_meta)
        self.noise_snr_range = noise_snr_range
        self.sampling_rate = sampling_rate
        self.noise_duration = int(noise_duration*self.sampling_rate)
        self.pointer = len(self.noise_meta)
        self.noise_order = []
    def reset_order(self):
        self.noise_order = np.random.permutation(len(self.noise_meta))
        self.pointer = 0
    def get_noise(self):
        if self.pointer >= len(self.noise_meta):
            self.reset_order()
        noise_path = self.noise_dir + self.noise_meta.iloc[self.noise_order[self.pointer]]['utterance_id']
        noise, rate = torchaudio.load(noise_path)
        noise = noise[0]
        noise = torchaudio.functional.resample(noise, rate, self.sampling_rate)
        if len(noise) < self.noise_duration:
            noise = torch.cat([noise, noise[:self.noise_duration-len(noise)]])
        elif len(noise) > self.noise_duration:
            noise = noise[:self.noise_duration]
        self.pointer += 1
        return noise

    def __call__(self, audio):
        noise = self.get_noise()
        snr = torch.tensor(np.random.uniform(*self.noise_snr_range))
        return torchaudio.functional.add_noise(audio, noise, snr)
    
    def add_noise_batch(self,audio):
        b = audio.shape[0]
        snr = torch.tensor(np.random.uniform(*self.noise_snr_range,size=b),device=audio.device)
        noise = self.get_noise()
        noise = repeat(noise,'n -> b n',b=b)
        return torchaudio.functional.add_noise(audio, noise, snr)

    
class LibriSpeechNoise:
    def __init__(
            self,
            cluster_list: Cluster,
            noise_snr_range: tuple,
            ):
        self.cluster_list = cluster_list
        self.noise_snr_range = noise_snr_range
    def __call__(self, audio):
        cluster = np.random.choice(self.cluster_list)
        noise, _ = cluster.get_val_item(np.random.randint(cluster.len_val_set()))
        noise = torchaudio.functional.resample(noise, 16000, 8000)
        snr = torch.tensor(np.random.uniform(*self.noise_snr_range))
        return torchaudio.functional.add_noise(audio, noise, snr)
# # # # Usage example
# noise_augmentation = WhamNoiseAugmentation(
#     'C:/Users/User/Documents/speech-diarization/wham_noise/wham_noise/tr/',
#     'C:/Users/User/Documents/speech-diarization/wham_noise/wham_noise/metadata/noise_meta_tr.csv',
#     (0, 20),
#     8000,
#     4
#     )

# a = torch.randn(32000)
# a = noise_augmentation(a)
# print(a.shape)
