from LibriSpeechDataset import *
from Ultils.data.AdjustBatch import BatchLibriMixLoader
# Initialize the dataset
dataset = LibriSpeech('speaker_and_speech.json')

# Define data transformation Augmentation
transform = Augmentation([
    DoNoThing(), DoNoThing()
    # AddNoise(),
    # AdjustVolumn(),
    # MaskOut()
], p = [0.5, 0.5])

# Initialize the LibriSpeechMixingLoader
dataloader = LibriSpeechMixingLoader(
    data=dataset,  # the dataset
    num=2, # number of speaker in 1 mixing audio
    sampleRate=16000, #the sample rate of audio
    audioLength=3,  # the length of each mixed audio in seconds
    speakerEmbeddingLength=1,  # the length of each speaker audio for speaker embedding in seconds
    batch_size=8,
    transform=transform  # apply the defined transformation
)

id = 0
# for o in dataloader:

#     id+=1
#     print(f"Mixing {id}:...")
#     print(o['mixing'][1:, :].shape)
#     torchaudio.save(f"mixaudio/Mixed_file1_{id}.wav", o['mixing'][0:1, :], o['mixing'].shape[1])
#     torchaudio.save(f"mixaudio/Mixed_file2_{id}.wav", o['mixing'][1:, :], o['mixing'].shape[1])
loader = BatchLibriMixLoader(dataloader, 26, 2)
for o in loader:
    # for k, v in o.items():
    #     print(k, v.shape)
    id+=1
    print(f"Mixing {id}:...")
    torchaudio.save(f"mixaudio/Mixed_file1_{id}.wav", o['vocal_sample'][1:2, :], 32000)

