import torch
import torchaudio
from pandas import DataFrame
from resemblyzer import VoiceEncoder
from torch.utils.data import Dataset
import os

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
            e = data['embedding']
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