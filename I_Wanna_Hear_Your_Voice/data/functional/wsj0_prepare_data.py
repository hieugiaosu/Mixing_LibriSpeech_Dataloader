import pandas as pd
import random
import os
import sys
from scipy.signal import resample_poly
from resemblyzer import VoiceEncoder
import soundfile as sf
from pathlib import Path
import numpy as np
FS_ORIG = 16000
class Wsj0Metadata():
    def __init__(self, 
                 filepath,
                 output_path,
                 n_src: int = 2,
                 ):
        self.filepath = filepath
        self.n_src = n_src
        self.output_path = output_path
        self.root = "/kaggle/input/wsj0-2mix/"
        self.embedding_model = VoiceEncoder(device = "cuda")
        self.audio_length = 64000

    def readDataFrame(self):
        header = [x for t in zip([f"s_{i}" for i in range(self.n_src)], [f"snr_{i}" for i in range(self.n_src)]) for x in t]
        mix_df = pd.read_csv(self.filepath, delimiter = " ", names = header, index_col = False)
        return mix_df
        
    def get_similar_files(self, filepath, all_files):
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        prefix = filename[:3]
        similar_files = [f for f in all_files if f.startswith(prefix) and f != filename]
        return [os.path.join(directory, f) for f in similar_files]
    
    def createMetadata(self):
        mix_df = self.readDataFrame()
        all_files = mix_df['s_0'].apply(os.path.basename).unique().tolist()

        # Create the new ref_audio_0 column
        mix_df['ref_audio_0'] = mix_df['s_0'].apply(lambda x: random.choice(self.get_similar_files(x, all_files)))

        # Create the new snr_ref column
        mix_df['snr_ref'] = mix_df['snr_0'] 
        
        #Create embedding column
        def emb(ref_path):
            ref_emb = sf.read(Path(self.root) / ref_path, dtype = "float32")[0]
            ref_emb = resample_poly(ref_emb, 8000, 16000)
            if len(ref_emb) < self.audio_length:
                ref_emb = np.hstack((ref_emb, np.zeros(self.audio_length - len(ref_emb))))
            else:
                start_index = random.randint(0, len(ref_emb) - self.audio_length)
                ref_emb = ref_emb[start_index: start_index + self.audio_length]
            ref_emb = self.embedding_model.embed_utterance(ref_emb)
            return ref_emb
        
        mix_df['ref_embedding'] = mix_df['ref_audio_0'].apply(emb)

        mix_df.to_csv(self.output_path, index = False)

if __name__ == "__main__":
    data1 = Wsj0Metadata("../metadata/mix_2_spk_cv.txt", "../metadata/mix_2_spk_cv.csv", 2)
    data1.createMetadata()
    data2 = Wsj0Metadata("../metadata/mix_2_spk_tt.txt", "../metadata/mix_2_spk_tt.csv", 2)
    data2.createMetadata()