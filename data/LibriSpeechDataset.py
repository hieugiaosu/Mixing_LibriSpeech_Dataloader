import torch
import torchaudio 
from torch.utils.data import Dataset
import pandas as pd
import os
import re
import math

"""
This function is to generate the dataframe structure of librispeech dataset
@param string path: the root path of librispeech
@param bool enumerateSpeaker:  if True the speaker will be enumerate by the order
os.listdir, else it will be the id of speaker. DEFAULT True
@return DataFrame: the DataFrame structure of dataset
"""

def getLibriSpeechDataFrame(
        path: str,
        enumerateSpeaker: bool = True,
        sample_rate: int = 16000,
        chunk_second_length: float = 2
    ) -> pd.DataFrame:
    try:
        count = 0
        data = []
        if path[-1] == '/':
            path = path[:-1]
        speakers = sorted(os.listdir(path), key=lambda x: int(x))
        chunk_duration = int(chunk_second_length * sample_rate)
        
        if enumerateSpeaker:
            speakerMap = {int(speaker): idx for idx, speaker in enumerate(speakers)}
        else:
            speakerMap = {int(speaker): int(speaker) for speaker in speakers}

        for speaker in speakers:
            speaker_idx = speakerMap[int(speaker)]
            speaker_path = os.path.join(path, speaker)
            
            for chapter in os.scandir(speaker_path):
                if chapter.is_dir():
                    chapter_path = chapter.path
                    audio_files = [f.path for f in os.scandir(chapter_path) if f.is_file() and not f.name.endswith('.txt')]
                    
                    for audio_file in audio_files:
                        audio_file = re.sub(r"\\","/",str(audio_file))
                        audio_info = torchaudio.info(audio_file)
                        audio_len = int(audio_info.num_frames)
                        count+=1
                        for i in range(0, audio_len - chunk_duration, chunk_duration):
                            data.append((speaker_idx, audio_file, i, i + chunk_duration))
        
        df = pd.DataFrame(data, columns=['speaker', 'audio_file', 'from_idx', 'to_idx'])
        print(f'Already process {count} audio file')
        return df, count
    
    except Exception as e:
        msg = f"Got {str(e)} in the execution. Maybe your librispeech file structure is not correct."
        raise RuntimeError(msg)


"""
LibriSpeechDataset: Custom Dataset for loading and processing the LibriSpeech dataset
@param DataFrame data: DataFrame containing speaker and audio_file columns
@param int sampleRate: Sampling rate for audio files. DEFAULT 16000
@param float maxAudioLength: Maximum length of audio clips (in seconds). DEFAULT 10
@param float windowSize
@param float hopLength
@param str paddingStrategy: Strategy for padding shorter audio clips. DEFAULT "zeros"
"""
class LibriSpeechDataset(Dataset):
    def __init__(self, data: pd.DataFrame,numFile:int, sampleRate=16000, 
                 maxAudioLength=10,paddingStrategy="zeros"):
        super().__init__()
        self.data = data
        self.sampleRate = sampleRate
        self.maxAudioLength = maxAudioLength
        self.maxAudioSamples = int(maxAudioLength*sampleRate)
        self.paddingStrategy = paddingStrategy
        self.cache = CacheData(numFile)
        self.len = len(self.data)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        speaker, audio_file, from_idx, to_idx = self.data.iloc[idx]
        waveform, sampleRate = self.cache.readFile(audio_file)
        if sampleRate != self.sampleRate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sampleRate,
                new_freq=self.sampleRate
                )
        padding = 0
        chunk = waveform[0,from_idx:to_idx].squeeze()
        if waveform.size(1) > self.maxAudioSamples:
            waveform = waveform[:, :self.maxAudioSamples]
        else:
            padding = self.maxAudioSamples - waveform.size(1)
            if self.paddingStrategy == "zeros":
                waveform = torch.nn.functional.pad(waveform, (padding, 0))
            elif self.paddingStrategy == "randn":
                padding = torch.randn(1,padding)/100
                waveform = torch.cat([padding,waveform],dim=1)
        waveform = waveform.squeeze()

        return waveform, chunk ,torch.tensor(speaker)
    
class CacheData:
    def __init__(self,numFile:int,minimumBatch:int = 20,maximumBatch:int=64) -> None:
        self.cacheSize = (2*numFile*math.log(2))/(minimumBatch*(minimumBatch-1))
        self.cacheSize = int(math.ceil(self.cacheSize))*maximumBatch

        self.hit = 0
        self.miss = 0

        self.keys = []
        self.cache = {}
    
    def readFile(self,fileName):
        audio = None 
        rate = None 
        try: 
            audio,rate = self.cache[fileName]
            self.hit += 1
        except:
            self.miss+=1
            audio, rate = torchaudio.load(fileName)
            if len(self.cache.keys()) < self.cacheSize:
                self.cache[fileName] = (audio,rate)
            else: 
                deleted_key = self.keys[0]
                self.cache.pop(deleted_key)
                self.keys.pop(0)
                self.cache[fileName] = (audio,rate)
            self.keys.append(fileName)
        return audio,rate
