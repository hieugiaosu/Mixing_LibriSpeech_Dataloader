import gradio as gr
import torch
import torchaudio
from resemblyzer import VoiceEncoder

from network.models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FilterBandTFGridnet(n_layers=5)
emb = VoiceEncoder(device=device)

def process_voice(voice):
    voice = voice.unsqueeze(0)
    voice = voice.float()
    voice = voice.to(device)
    return voice

def load_voice(voice_path):
    voice, _ = torchaudio.load(voice_path)
    return voice

def main(mixed_voice_path, clean_voice_path):
    mixed_voice = load_voice(mixed_voice_path)
    clean_voice = load_voice(clean_voice_path)
    if not model:
        return None
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(
            process_voice(torch.tensor(mixed_voice)),
            process_voice(emb.embed_utterance(clean_voice.numpy())),
        )

def load_checkpoint(filepath):
    checkpoint = torch.load(
        filepath,
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(checkpoint)

samples = {
    '1': ['1.flac', '2.flac'],
}

with gr.Blocks() as demo:
    # load_checkpoint('checkpoint/filterband.pth')
    with gr.Row():
        with gr.Column():
            mixed_voice = gr.Audio(label='Mixed voice', type='filepath')
            clean_voice = gr.Audio(label='Clean voice', type='filepath')

            choices = list(samples.keys())
            dropdown = gr.Dropdown(choices=choices, label='Samples')
            dropdown.input(
                fn=lambda x: samples[x],
                inputs=[dropdown],
                outputs=[mixed_voice, clean_voice],
            )
        with gr.Column():
            sep_voice = gr.Audio(label="Separate Voice")
            btn = gr.Button("Separate voices", size='sm')
            btn.click(main, inputs=[mixed_voice, clean_voice], outputs=sep_voice)

demo.launch()

