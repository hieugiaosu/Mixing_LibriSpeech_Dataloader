import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import numpy as np
import json
from multiprocessing import Semaphore
import warnings

class Singleton(type):
    _instances = {}
    _semaphore = Semaphore()
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances or cls._instances[cls] is None:
            with cls._semaphore:
                if cls not in cls._instances or cls._instances[cls] is None:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
                else: 
                    warnings.warn("Warning: This class is using singleton design. It mean that your object creation using the old instance")
        else:
            warnings.warn("Warning: This class is using singleton design. It mean that your object creation using the old instance")
        return cls._instances[cls]

class LibriSpeech(metaclass=Singleton):
    def __init__(self,data,sample_rate=16000,return_type='pt'):
        assert return_type in ['pt','np'], "invalid return type"
        with open(data,mode='r') as f: 
            self.data = json.load(f)
        if 'speaker' in self.data.keys():
            self.ignore_chapter = False
        else: 
            self.ignore_chapter = True 
        self.sample_rate = sample_rate 
        self.return_type = return_type
    def changeDataFile(self,data):
        with open(data,mode='r') as f: 
            self.data = json.load(f)
        if 'speaker' in self.data.keys():
            self.ignore_chapter = False
        else: 
            self.ignore_chapter = True 
    def __getitem__(self,idx:tuple):
        try:
            if len(idx)==2:
                speaker,segment = idx 
                chapter=None
            else:
                speaker,segment,chapter=idx
            if not self.ignore_chapter:
                assert chapter is not None, "your file structure need to use chapter"
                url = self.data['speakers'][str(speaker)]['chapters'][str(chapter)]['segments'][str(segment)]
            else: 
                url = self.data[str(speaker)][str(segment)]
            waveform,rate = torchaudio.load(url)
            if rate != self.sample_rate:
                waveform = F.resample(waveform, rate, self.sample_rate)
            if self.return_type == 'np':
                waveform = waveform.numpy()
            return waveform
        except Exception as e: 
            print(e)
            return torch.zeros((1,16000))
    def getAudio(self,speaker,segment,chapter=None):
        return self.__getitem__((speaker,segment,chapter))
    def getSpeakerList(self):
        if not self.ignore_chapter:
            return list(map(int,self.data['spearkers'].keys()))
        else:
            return list(map(int,self.data.keys()))
    def getNumSegmentsBySpeaker(self,speaker):
        assert self.ignore_chapter,"this method does not support for your file"
        return len(list(self.data[str(speaker)].keys()))
    def getRandomSegment(self,length,exceptSpeaker=None):
        assert self.ignore_chapter,"this method does not support for your file"
        speakerList = self.getSpeakerList()
        if exceptSpeaker:
            speakerList = [speaker for speaker in speakerList if speaker not in exceptSpeaker]
        speaker = np.random.choice(speakerList)
        audio = self.__getitem__((speaker,
                                 np.random.randint(self.getNumSegmentsBySpeaker(speaker))))
        limitStartSegment = audio.shape[1] - int(self.sample_rate*length)
        if limitStartSegment <= 0:
            return self.getRandomSegment(length,exceptSpeaker)
        startSegment = np.random.randint(0,limitStartSegment)
        return audio[:,startSegment:startSegment+int(length*self.sample_rate)]
    
    def getRandomSegmentBySpeaker(self,length,speaker):
        assert self.ignore_chapter,"this method does not support for your file"
        audio = self.__getitem__((speaker,
                                 np.random.randint(self.getNumSegmentsBySpeaker(speaker))))
        limitStartSegment = audio.shape[1] - int(self.sample_rate*length)
        if limitStartSegment <= 0:
            return self.getRandomSegmentBySpeaker(length,speaker)
        startSegment = np.random.randint(0,limitStartSegment)
        return audio[:,startSegment:startSegment+int(length*self.sample_rate)]
class DoNoThing(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class AdjustVolumn(nn.Module):
    def __init__(self,func=torch.cos,k=1e5):
        super().__init__()
        self.func = func
        self.k = k 
    def forward(self,x):
        scale = self.func(torch.arange(x.shape[1])/self.k)**2
        return x*scale
class AddNoise(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x + torch.randn_like(x)/1000

class MaskOut(nn.Module):
    def __init__(self,p=0.3,maxlen=16000):
        super().__init__()
        self.p = p 
        self.maxlen = maxlen
    def forward(self,x):
        if np.random.uniform(0,1)>self.p:
            return x
        num_voice = x.shape[0]
        voice = torch.tensor(np.random.permutation(num_voice)[:np.random.randint(1,num_voice+1)]).long()
        masklen = int(np.random.randint(int(self.maxlen/2),self.maxlen+1))
        start = int(np.random.randint(0,x.shape[1]-masklen))
        mask = torch.ones_like(x)
        mask[voice,start:start+masklen] = 0 
        mask = mask.float()
        return x*mask
class Augmentation:
    def __init__(self,transform:list,p:list[float]=None):
        self.transform = transform
        if p is None: 
            p = [1/len(transform)]*len(transform)
        self.p = p
    def __call__(self,audio):
        transform = np.random.choice(self.transform,p=self.p)
        return transform(audio)

class LibriSpeechMixingLoader:
    def __init__(self,data:LibriSpeech,num,sampleRate,
                 audioLength,speakerEmbeddingLength,
                 batch_size=1,
                 transform=Augmentation([DoNoThing()])):
        self.data = data 
        self.sampleRate = sampleRate
        self.audioLength = audioLength
        self.speakerEmbeddingLength = speakerEmbeddingLength
        self.num = num 
        self.batch = batch_size
        self.transform = transform
        
    def __iter__(self):
        # warnings.warn("notice that")
        self.permutation = np.random.permutation(self.data.getSpeakerList())
        self.permutation = self.permutation[:len(self.permutation)-len(self.permutation)%self.num]
        self.permutation = np.array(self.permutation).reshape((-1,self.num))
        self.audioStore = {}
        self.sampleEmbed = {}
        self.segmentOrder = []
        self.idx = 0
        self.windowStart = 0
        self.segmentIdx = 0
        self.reachend = False
        return self
    def __getGroupData(self,startwindow):
        if startwindow > self.audioStore.shape[1] - int(self.sampleRate*self.audioLength):
            self.reachend = True
            startwindow = self.audioStore.shape[1] - int(self.sampleRate*self.audioLength)
        audio = self.audioStore[:,startwindow:startwindow+int(self.sampleRate*self.audioLength)]
        audio = self.transform(audio)
        return {"mixing": torch.mean(audio,dim=0),"audio":audio}


    def __next__(self):
        if self.idx == len(self.permutation): raise StopIteration()
        ### now it is new speaker
        if self.segmentIdx == 0:
            self.segmentOrder = [self.data.getNumSegmentsBySpeaker(i) for i in self.permutation[self.idx]]
            self.segmentOrder = list(map(np.random.permutation,self.segmentOrder))
            lim = min(map(len,self.segmentOrder))
            self.segmentOrder = list(map(lambda x: x[:lim],self.segmentOrder))
            self.segmentOrder = np.stack(self.segmentOrder)
            self.lim = lim
        ## now it is new segment
        if self.windowStart == 0:
            self.audioStore = list(map(lambda x: self.data[x],
                                               zip(self.permutation[self.idx],self.segmentOrder[:,self.segmentIdx])))
            maxlen = max(map(lambda x: len(x[0]),self.audioStore))
            self.audioStore = list(map(
                lambda x: torch.cat(
                    [x,torch.zeros((1,maxlen-len(x[0])))],
                    dim=1
                    ), self.audioStore))
            self.audioStore = torch.cat(self.audioStore,dim=0)
            self.sampleEmbed = list(map(lambda x: self.data.getRandomSegmentBySpeaker(self.speakerEmbeddingLength,int(x)),
                                        self.permutation[self.idx]))
            self.sampleEmbed = torch.cat(self.sampleEmbed,dim=0)
        data = [self.__getGroupData(self.windowStart+i*self.sampleRate*int(self.audioLength*0.5)) for i in range(self.batch) if not self.reachend]
        o = {}
        o['mixing'] = torch.stack([i['mixing'] for i in data])
        o['audio'] = torch.cat([i['audio'] for i in data],dim=0)
        o['vocal_sample'] = torch.cat([self.sampleEmbed]*len(o['mixing']),dim=0)
        o['speaker_id'] = torch.cat([torch.tensor(self.permutation[self.idx])]*len(o['mixing']),dim=0).long()
        self.windowStart=self.windowStart+self.batch*self.sampleRate*int(self.audioLength*0.5)
        if self.windowStart >= self.audioStore.shape[1]  - int(self.sampleRate*self.audioLength)*2: ##*2 for no reason :))
            self.reachend = True
        if self.reachend:
            self.segmentIdx+=1
            self.windowStart = 0
            self.reachend = False
        if self.segmentIdx == self.lim:
            self.idx+=1
            self.segmentIdx = 0
        return o