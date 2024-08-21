import torch
import torchaudio
from pandas import DataFrame
from resemblyzer import VoiceEncoder
from torch.utils.data import Dataset
import os
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path
import numpy as np
import random
FS_ORIG = 16000
class CacheTensor:
    def __init__(self,cacheSize:int,miss_handler) -> None:
        self.cacheSize = cacheSize
        self.miss_handler = miss_handler

        self.keys = []
        self.cache = {}
    
    def __readFile(self,fileName):
        data = None
        try: 
            data = self.cache[fileName]
        except:
            data = self.miss_handler(fileName)
            if len(self.cache.keys()) < self.cacheSize:
                self.cache[fileName] = data
            else: 
                deleted_key = self.keys[0]
                self.cache.pop(deleted_key)
                self.keys.pop(0)
                self.cache[fileName] = data
            self.keys.append(fileName)
        return data
    
    def __call__(self, fileName):
        return self.__readFile(fileName)
    
class LibriSpeech2MixDataset(Dataset):
    def __init__(
            self, 
            df: DataFrame,
            root = '', 
            sample_rate:int = 8000,
            using_cache = False,
            cache_size = 1,
            device = 'cuda'
            ):
        super().__init__()
        self.data = df
        self.sample_rate = sample_rate
        self.root = root
        self.device = device
        if not using_cache or cache_size == 1:
            self.file_source = torchaudio.load
        else: 
            self.file_source = CacheTensor(cache_size,torchaudio.load)
        if 'embedding' not in df.columns:
            self.use_encoder = True
            self.embedding_model = VoiceEncoder(device = device)
        else:
            self.use_encoder = False
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        audio_file = os.path.join(self.root, data['audio_file'])
        from_idx = data['from_idx']
        to_idx = data['to_idx']
        mix_audio_file = os.path.join(self.root, data['mix_audio_file'])
        mix_from_idx = data['mix_from_idx']
        mix_to_idx = data['mix_to_idx']
        ref_audio_file = os.path.join(self.root, data['ref_audio_file'])
        ref_from_idx = data['ref_from_idx']
        ref_to_idx = data['ref_to_idx']
        
        
        first_waveform,rate = self.file_source(audio_file)
        first_waveform = first_waveform.squeeze()[from_idx:to_idx]
        if self.use_encoder:
            e = torch.tensor(self.embedding_model.embed_utterance(first_waveform.numpy())).float().cpu()
        else:
            e = eval(data['embedding'])
            e = torch.tensor(e).float()

        second_waveform,rate = self.file_source(mix_audio_file)
        second_waveform = second_waveform.squeeze()[mix_from_idx:mix_to_idx]
             
        ref_waveform,rate = self.file_source(ref_audio_file)
        ref_waveform = ref_waveform.squeeze()[ref_from_idx:ref_to_idx]
        if rate != self.sample_rate:
            first_waveform = torchaudio.functional.resample(first_waveform,rate,self.sample_rate)
            ref_waveform = torchaudio.functional.resample(ref_waveform,rate,self.sample_rate)
            second_waveform = torchaudio.functional.resample(second_waveform,rate,self.sample_rate)
        
        mix_waveform = torchaudio.functional.add_noise(first_waveform,second_waveform,torch.tensor(1))
        return {"mix":mix_waveform, "src0": first_waveform, "src1":second_waveform, "ref0":ref_waveform, "emb0": e}
    



class Wsj02MixDataset(Dataset):
    def __init__(
            self, 
            df: DataFrame,
            root = '', 
            sample_rate:int = 8000,
            using_cache = False,
            cache_size = 1,
            device = 'cuda',
            mode = "max",
            n_srcs = 2,
            chunk_duration = 4,
            ):
        super().__init__()
        self.data = df
        self.sample_rate = sample_rate
        self.root = root
        self.device = device
        self.n_srcs = n_srcs
        self.mode = mode
        self.chunk_duration = chunk_duration
        self.audio_length = self.chunk_duration * FS_ORIG
        if 'embedding' not in df.columns:
            self.use_encoder = True
            self.embedding_model = VoiceEncoder(device = device)
        else:
            self.use_encoder = False
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        sources = [sf.read(Path(self.root) / data[f"s_{i}"], dtype = "float32")[0] for i in range(self.n_srcs)]
        snrs = [data[f"snr_{i}"] for i in range(self.n_srcs)]
        # ref_audio = sf.read(Path(self.root) / data["ref_audio_0"], dtype = "float32")[0]

        resampled_sources = [resample_poly(s, self.sample_rate, FS_ORIG) for s in sources]
        # resampled_ref = resample_poly(ref_audio, self.sample_rate, FS_ORIG)

        def padding(sample):
            if len(sample) < self.audio_length:
                sample_padding = np.hstack((sample, np.zeros(self.audio_length - len(sample))))
                return sample_padding
            start_index = random.randint(0, len(sample) - self.audio_length)
            sample_padding = sample[start_index: start_index + self.audio_length]
            return sample_padding
        
        # min_len, max_len = min([len(s) for s in resampled_sources]), max([len(s) for s in resampled_sources])
        # padded_sources = [np.hstack((s, np.zeros(max_len - len(s)))) for s in resampled_sources]
        padded_sources = list(map(padding, resampled_sources))
        # resampled_ref = padding(resampled_ref)
        
        # padded_ref = np.hstack((resampled_ref, np.zeros(max_len - len(resampled_ref))))

        activlev_scales = [np.sqrt(np.mean(s**2)) for s in resampled_sources]
        scaled_sources = [s / np.sqrt(scale) * 10 ** (x/20) for s, scale, x in zip(padded_sources, activlev_scales, snrs)]

        
        sources_np = np.stack(scaled_sources, axis=0)
        mix_np = np.sum(sources_np, axis=0)

   
        # e = torch.tensor(self.embedding_model.embed_utterance(resampled_ref)).float().cpu()

        ref_embedding = data['ref_embedidng'].split(" ")
        print(len(ref_embedding))
        e = eval(ref_embedding)
        e = torch.tensor(e).float()

        if self.mode == "max":
            gain = np.max([1., np.max(np.abs(mix_np)), np.max(np.abs(sources_np))]) / 0.9
            mix_np_max = mix_np / gain
            sources_np_max = sources_np / gain
            return {"mix": mix_np_max, "src0": sources_np_max[0], "src1": sources_np_max[1], "emb0": e}

        # if self.mode == "min":
        #     sources_np = sources_np[:,:min_len]
        #     mix_np = mix_np[:min_len]
        #     gain = np.max([1., np.max(np.abs(mix_np)), np.max(np.abs(sources_np))]) / 0.9
        #     mix_np /= gain
        #     sources_np /= gain
        #     return {"mix": mix_np[:min_len], "src0": sources_np[0][:min_len], "src1": sources_np[1][:min_len], "ref0": ref_audio, "emb0": e}
        # mix_waveform = torchaudio.functional.add_noise(first_waveform,second_waveform,torch.tensor(1))
  
    
