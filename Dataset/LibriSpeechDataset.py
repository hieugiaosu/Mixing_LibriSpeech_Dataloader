import torch
import torchaudio 
from torch.utils.data import Dataset
import pandas as pd
import os


"""
This function is to generate the dataframe structure of librispeech dataset
@param string path: the root path of librispeech
@param bool enumerateSpeaker:  if True the speaker will be enumerate by the order
os.listdir, else it will be the id of speaker. DEFAULT True
@return DataFrame: the DataFrame structure of dataset
"""

def getLibriSpeechDataFrame(path: str,enumerateSpeaker:bool = True) -> pd.DataFrame:
    try:
        df = pd.DataFrame([],columns=['speaker','audio_file'])
        if path[-1] == '/':
            path = path[:-1]
        speakers = os.listdir(path)
        speakers = sorted(speakers,key = lambda x: int(x))
        if enumerateSpeaker:
            speakerMap = dict({int(speaker):idx for idx,speaker in enumerate(speakers)})
        else:
            # This implement is not efficient about the memmory storage but it is not 
            # affect too much, i implement like this for the easy understand code later
            speakerMap = dict({int(speaker):int(speaker) for speaker in speakers})
        for speaker in speakers:
            audioFile = list([])
            for chapter in os.listdir(f"{path}/{speaker}"):
                audiofileList = list(filter(lambda x: 'txt' not in x,os.listdir(f'{path}/{speaker}/{chapter}')))
                audiofileList = list(map(lambda x: (speakerMap[int(speaker)],f"{path}/{speaker}/{chapter}/{x}"),audiofileList))
                audioFile += audiofileList
            newRows = pd.DataFrame(audioFile,columns=['speaker','audio_file'])
            df = pd.concat([df,newRows],axis=0)
        return df
    except Exception as e:
        msg = f"Got {str(e)} in the execution\nMaybe your librispeech file structure is not correct"
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
    def __init__(self, data: pd.DataFrame, sampleRate=16000, 
                 maxAudioLength=10, windowSize = 2, 
                 hopLength = 0.4 ,paddingStrategy="zeros"):
        super().__init__()
        self.data = data
        self.sampleRate = sampleRate
        self.maxAudioLength = maxAudioLength
        self.paddingStrategy = paddingStrategy
        self.maxAudioSamples = int(self.sampleRate * self.maxAudioLength)
        self.windowSize = int(windowSize*self.sampleRate)
        self.hopLength = int(hopLength*self.sampleRate)
        self.numWindowEachAudio = int(maxAudioLength/hopLength)
        self.len = self.numWindowEachAudio*len(self.data)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        audioIdx = idx // self.numWindowEachAudio
        windowIdx = idx % self.numWindowEachAudio

        speaker, audioFile = self.data.iloc[audioIdx]
        waveform, sampleRate = torchaudio.load(audioFile)
        if sampleRate != self.sampleRate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sampleRate,
                new_freq=self.sampleRate
                )
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
        window = waveform[windowIdx*self.hopLength:windowIdx*self.hopLength+self.windowSize]
        return window,waveform, speaker