from LibriSpeechDataset import *

# Initialize the dataset
dataset = LibriSpeech('speaker_and_speech.json')

# Define data transformation Augmentation
transform = Augmentation([
    DoNoThing(),
    AddNoise(),
    AdjustVolumn(),
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
        # print(k, v.shape)
        print('.')