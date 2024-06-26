 # LibriSpeech DataLoader For Pytorch

This repository contains a custom dataloader for the LibriSpeech dataset. To utilize this dataloader, follow these simple steps:

1. Visit [Open Speech and Language Resources](https://www.openslr.org/12) to access the dataset.
2. Download and unzip the dataset within the designated folder.

## Metadata File
In this repository, a pre-created metadata file is available for the `train-clean-100` subset. You can conveniently download `train-clean-100` and utilize `speaker_and_speech.json` as the metadata file.

## Quick Start Guide
To swiftly get started with dataloader, use the following code snippets:

```python
from LibriSpeechDataset import *

# Initialize the dataset
dataset = LibriSpeech('speaker_and_speech.json')

# Define data transformation Augmentation
transform = Augmentation([
    DoNothing(),
    AddNoise(),
    AdjustVolume(),
    MaskOut()
])

# Initialize the LibriSpeechMixingLoader
dataloader = LibriSpeechMixingLoader(
    data=dataset,  # the dataset
    num=4, # number of speaker in 1 mixing audio
    sampleRate=16000, #the sample rate of audio
    audioLength=4,  # the length of each mixed audio in seconds
    speakerEmbeddingLength=2,  # the length of each speaker audio for speaker embedding in seconds
    batch_size=2,
    transform=transform  # apply the defined transformation
)

# Iterate over the dataloader
for o in dataloader:
    for k, v in o.items():
        print(k, v.shape)
```

For training:

```python
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
```
Ensure to replace the placeholders and adjust parameters as needed before running the code.  