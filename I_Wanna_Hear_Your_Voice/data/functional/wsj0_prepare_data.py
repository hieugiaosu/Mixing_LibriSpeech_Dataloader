import pandas as pd
import random
import os
import sys
# sys.path.append("/home/tanio/nam3/NCKH/Mixing_LibriSpeech_Dataloader/I_Wanna_Hear_Your_Voice/data")
# print(sys.path)
class Wsj0Metadata():
    def __init__(self, 
                 filepath,
                 output_path,
                 n_src: int = 2,
                 ):
        self.filepath = filepath
        self.n_src = n_src
        self.output_path = output_path
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
        
        mix_df.to_csv(self.output_path, index = False)

if __name__ == "__main__":
    data = Wsj0Metadata("../metadata/mix_2_spk_tt.txt", "../metadata/mix_2_spk_tt.csv", 2)
    data.createMetadata()