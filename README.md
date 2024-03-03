 # LibriSpeech DataLoader

This repository contains a custom dataloader for the LibriSpeech dataset. To utilize this dataloader, follow these simple steps:

1. Visit [Open Speech and Language Resources](https://www.openslr.org/12) to access the dataset.
2. Download and unzip the dataset within the designated folder.

## Metadata File
In this repository, a pre-created metadata file is available for the `train-clean-100` subset. You can conveniently download `train-clean-100` and utilize `speaker_and_speech.json` as the metadata file.

## Quick Start Guide
To swiftly get started, use the following code snippets:

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

Ensure to replace the placeholders and adjust parameters as needed before running the code.  