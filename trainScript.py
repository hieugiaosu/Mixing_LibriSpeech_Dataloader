import torch
from model.speaker_embedding.Module import SpeakerEmbedding, InputConfig
import model.seperate_speech.Module as sep
from Ultils.training.Pipeline import TrainPipeline
from LibriSpeechDataset import *
import sys
sys.path.append("./model/speaker_embedding/model-weights/")
dataset = LibriSpeech('speaker_and_speech.json')
transform = Augmentation([
    DoNoThing(),
    AddNoise(),
    AdjustVolumn(),
    MaskOut(p=0.6)
],p = [0.1,0.2,0.3,0.4])
datasetLibri = LibriSpeechMixingLoader(
    data=dataset,num=4,sampleRate=16000,audioLength=2,
    speakerEmbeddingLength=0.5,batch_size=8,transform=transform
)
e_inputConfig = InputConfig(**InputConfig.defaultInitArgs())
s_inputConfig = sep.AEInputConfigAfterEmbedding()
e_model = SpeakerEmbedding(e_inputConfig.getAudioShape(),e_inputConfig.getMelShape())
e_model.load_state_dict(torch.load('model/speaker_embedding/model-weights/speaker_embedding_model.pth',map_location='cpu'))
s_model = sep.AEBaseModel(32000)
pipeline = TrainPipeline(e_model,"speaker_embedding","model/speaker_embedding/model-weights/",
                         "model/speaker_embedding/train_log/",e_inputConfig,s_model,
                         'speech_sep','model/seperate_speech/model-weight/',
                         'model/seperate_speech/train_log/',s_inputConfig,12,100,datasetLibri,
                         state=14022003,using_gpu=True,multi_gpu=True
                         )
pipeline.train()