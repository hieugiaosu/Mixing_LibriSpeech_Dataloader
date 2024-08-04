from torch.utils.data import Dataset, DataLoader
from .cluster import Cluster
from typing import List
import torchaudio
import numpy as np
import torch

class TrainDatasetWithCluster(Dataset):
    def __init__(self,clusters:List[Cluster], embedding_model,augmentation = None,num_speaker_per_cluster:int=6, sampling_rate = 8000, log = True) -> None:
        super().__init__()

        self.clusters = clusters
        self.log = log
        self.cluster_size = [0]*(len(self.clusters)+1)
        self.num_speaker_per_cluster = num_speaker_per_cluster
        self.reset()
        self.embedding_model = embedding_model
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation
        

    def reset(self):
        for i in range(len(self.clusters)):
            self.clusters[i].reset(self.num_speaker_per_cluster,self.log)
            self.cluster_size[i+1] = len(self.clusters[i]) + self.cluster_size[i]
        
    def __len__(self): return self.cluster_size[-1]

    def __getitem__(self,idx):
        cluster = self.clusters[0]
        for i,lim in enumerate(self.cluster_size):
            if idx < lim:
                cluster = self.clusters[i - 1]
                idx -= self.cluster_size[i - 1]
                break

        speaker, chunk_idx = cluster[idx]

        audio, ref = cluster.read_file(speaker,chunk_idx,True)
        e = self.embedding_model(ref)

        audio = torchaudio.functional.resample(audio,16000,self.sampling_rate)

        second_audio = self.clusters[np.random.randint(0,len(self.clusters))].get_mix_for_speaker(int(speaker),False)
        second_audio = torchaudio.functional.resample(second_audio,16000,self.sampling_rate)
        snr_rate = torch.randint(0,5,(1,))[0]
        mix = torchaudio.functional.add_noise(audio,second_audio,snr_rate)
        if self.augmentation is not None:
            mix = self.augmentation(mix)
        
        return {"mix":mix,"src0":audio,"emb0":e}
    
class ValDatasetWithCluster(Dataset):
    def __init__(self,clusters:List[Cluster], embedding_model,augmentation = None,num_speaker_per_cluster:int=6, sampling_rate = 8000) -> None:
        super().__init__()

        self.clusters = clusters
        self.cluster_size = [0]*(len(self.clusters)+1)
        self.num_speaker_per_cluster = num_speaker_per_cluster
        self.embedding_model = embedding_model
        self.sampling_rate = sampling_rate
        self.augmentation = augmentation

        for i in range(len(self.clusters)):
            self.cluster_size[i+1] = self.clusters[i].len_val_set() + self.cluster_size[i]
        
    def __len__(self): return self.cluster_size[-1]

    def __getitem__(self,idx):
        cluster = self.clusters[0]
        for i,lim in enumerate(self.cluster_size):
            if idx < lim:
                cluster = self.clusters[i - 1]
                idx -= self.cluster_size[i - 1]
                break
        audio, speaker_id = cluster.get_val_item(idx)
        e = self.embedding_model(audio)

        audio = torchaudio.functional.resample(audio,16000,self.sampling_rate)
        second_audio = self.clusters[np.random.randint(0,len(self.clusters))].get_mix_for_speaker(int(speaker_id),True)
        second_audio = torchaudio.functional.resample(second_audio,16000,self.sampling_rate)
        snr_rate = torch.randint(0,5,(1,))[0]
        
        mix = torchaudio.functional.add_noise(audio,second_audio,snr_rate)
        if self.augmentation is not None:
            mix = self.augmentation(mix)
        
        return {"mix":mix,"src0":audio,"emb0":e}

class TrainDataLoaderWithCluster(DataLoader):
    def __iter__(self):
        self._iterator = super().__iter__()
        return self
    
    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self.dataset.reset()
            self._iterator = super().__iter__()
            raise StopIteration