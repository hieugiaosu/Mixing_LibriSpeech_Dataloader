import torch 
import torch.nn as nn  
import torch.nn.functional as F
import torchaudio.transforms as T 
import librosa

class InputConfig:
    @staticmethod
    def defaultInitArgs(sampleRate=16000,sampleLength=0.5):
        return {
            "sampleRate":sampleRate, "sampleLength":sampleLength,
            "n_fft": 2048, "n_mels": 256,
            "hop_length":512,"mel_scale": "htk"
        }
    def __init__(self,sampleRate=16000,sampleLength=0.5,**args):
        self.sampleRate = sampleRate
        self.sampleLength = sampleLength
        self.mel = T.MelSpectrogram(sample_rate=sampleRate,**args)
        melShape = self.mel(torch.randn(1,int(self.sampleRate*self.sampleLength))).shape
        self.melShape = (melShape[-1],melShape[-2])
    def __call__(self, audio) -> torch.Any:
        mel = self.mel(audio).float()
        mel = torch.transpose(mel,1,2)
        return {'audio':audio,'mel':mel}
    def getAudioShape(self):
        return (int(self.sampleRate*self.sampleLength),)
    def getMelShape(self):
        return self.melShape
class EncoderBlock(nn.Module):
    def __init__(self,inputShape,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputShape = inputShape
        self.qextract = nn.Conv1d(inputShape[0],inputShape[0],25,padding='same')
        self.kextract = nn.Conv1d(inputShape[0],inputShape[0],25,padding='same')
        self.vextract = nn.Conv1d(inputShape[0],inputShape[0],25,padding='same')
        self.attention = nn.MultiheadAttention(inputShape[1],1,batch_first = True)
        self.normalize = nn.LayerNorm(self.inputShape)
        self.silu = nn.SiLU()
    def forward(self,x):
        q = self.qextract(x)
        k = self.kextract(x)
        v = self.vextract(x)
        att = self.attention(q,k,v)[0]
        att = self.silu(att)
        o = self.normalize(x+att)
        return o
class DecoderBlock(nn.Module):
    def __init__(self,inputShape,encoderInputDim,*args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.inputShape = inputShape
        # self.outputDim = outputDim
        self.encoderInputDim = encoderInputDim
        self.inputEncode = EncoderBlock(inputShape)
        self.encoderTransform = nn.Sequential(
            nn.Linear(encoderInputDim,inputShape[1]),
            nn.SiLU()
        )
        self.attention = nn.MultiheadAttention(inputShape[1],1,batch_first = True)
        self.normalize = nn.LayerNorm(self.inputShape)
        self.silu = nn.SiLU()
        self.lastTransform = nn.Sequential(
            nn.Conv1d(inputShape[0],inputShape[0],25,padding='same'),
            nn.SiLU()
        )
    def forward(self,x,encoderInput):
        q = self.inputEncode(x)
        k = self.encoderTransform(encoderInput)
        att = self.attention(q,k,k)[0]
        att = self.silu(att)
        o = self.normalize(q+att)
        o = self.lastTransform(o)
        o = (o + x)/2
        return o
class MelEncoder(nn.Module):
    def __init__(self,melInputShape,outputDim,encoderLayerNum=1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputShape = melInputShape
        self.outputDim = outputDim
        self.inputTransform = nn.Sequential(
            nn.Linear(melInputShape[1],melInputShape[1]),
            nn.GELU(),
            nn.LayerNorm(self.inputShape)
        )
        self.encoder = nn.Sequential(
            EncoderBlock(melInputShape)
        )
        for _ in range(encoderLayerNum-1):
            self.encoder.append(EncoderBlock(melInputShape))
        self.outputTransform = nn.Sequential(
            nn.Linear(self.inputShape[1],outputDim),
            nn.Tanh()
        )
    def forward(self,x):
        o = self.inputTransform(x)
        o = self.encoder(o)
        o = self.outputTransform(o)
        return o
class SpeakerEmbedding(nn.Module):
    def __init__(self,audioInputShape,melInputShape,encoderLayers=1,decoderLayers = 3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.audioLength =  audioInputShape[0]

        windowSize = 513
        stride = 32 
        currOut = (self.audioLength-(windowSize-1)-1)/stride 
        padding = (int(currOut)+1 - currOut)*stride/2
        padding = int(padding)+1
        currOut = int((self.audioLength+2*padding-(windowSize-1)-1)/stride) +1
        currOut -= 64

        self.melEncoder = MelEncoder(melInputShape,512,encoderLayers)
        self.inputTransform = nn.Sequential(
            nn.Conv1d(1,128,windowSize,stride,padding),
            nn.GELU(),
            nn.Conv1d(128,256,33),
            nn.GELU(),
            nn.Conv1d(256,512,33),
            nn.GELU()
        )
        firstlayerShape =  (currOut,512)

        self.decoderBlock = nn.ModuleList([DecoderBlock(firstlayerShape,512)]*decoderLayers)

        self.lastTransform = nn.Conv1d(currOut,currOut,25,padding='same')
        self.tanh = nn.Tanh()
    def forward(self,audio,mel):
        encoder_o = self.melEncoder(mel)
        audio = audio.unsqueeze(1)
        o = self.inputTransform(audio)
        o = torch.transpose(o,1,2)
        for decoder in self.decoderBlock:
            o = decoder(o,encoder_o)
        
        o = self.lastTransform(o)
        o = torch.transpose(o,1,2)
        o = o.mean(dim=2)
        o = self.tanh(o)
        return o
class EmbeddingLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,emb,label):
        n = label.shape[0]
        corr = torch.triu((label.unsqueeze(0) == label.unsqueeze(1)),diagonal=1)
        e = F.normalize(emb, p=2, dim=1)
        similarity = torch.abs(e@e.T)
        similarity[corr] = 1-similarity[corr] 
        similarity = torch.sum(torch.triu(similarity,diagonal=1))/int(n*n/2-n/2)
        return similarity