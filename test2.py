import torchaudio
import torch

# Load the FLAC files
flac_file1 = "/home/tanio/nam3/NCKH/Mixing_LibriSpeech_Dataloader/train-clean-100/LibriSpeech/train-clean-100/19/198/19-198-0010.flac"
flac_file2 = "/home/tanio/nam3/NCKH/Mixing_LibriSpeech_Dataloader/train-clean-100/LibriSpeech/train-clean-100/26/495/26-495-0012.flac"

waveform1, sample_rate1 = torchaudio.load(flac_file1)
waveform2, sample_rate2 = torchaudio.load(flac_file2)

# Resample if necessary (optional)
# if sample_rate1 != sample_rate2:
#     print("Resampling...")
#     resampler = torchaudio.transforms.Resample(sample_rate2, sample_rate1)
#     waveform2 = resampler(waveform2)

print("Resampling waveform1...")
resampler1 = torchaudio.transforms.Resample(sample_rate1, 8000)
waveform1 = resampler1(waveform1)

print("Resampling waveform2...")
resampler2 = torchaudio.transforms.Resample(sample_rate2, 8000)
waveform2 = resampler2(waveform2)


min_length = min(waveform1.size(1), waveform2.size(1))
waveform1 = waveform1[:, :min_length]
waveform2 = waveform2[:, :min_length]

# Mix the waveforms
mixed_waveform = waveform1 + waveform2

# Save the mixed waveform as WAV file
mixed_file = "mixed_output.wav"
torchaudio.save(mixed_file, mixed_waveform, sample_rate1)

print("Mixed audio saved as:", mixed_file)
