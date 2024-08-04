import json
import torchaudio
import pandas as pd
import numpy as np
import os

class Cluster:
    def __np_softmax(self, data):
        """
        Apply softmax function to a numpy array.
        
        Parameters:
        - data: List of numerical values.
        
        Returns:
        - Softmax-distributed values of the input data.
        """
        p = np.array(data)
        e = np.exp(p)
        distribute = e / e.sum(0)
        return distribute

    def __init__(self, cluster_id: int, distance_meta_data: str, root_path: str, data=None, cluster_meta=None, sampling_rate=16000, val_size=0):
        """
        Initialize the Cluster object.
        
        Parameters:
        - cluster_id: Unique identifier for the cluster.
        - distance_meta_data: Path to CSV file containing distance metadata.
        - root_path: Root directory path for audio files.
        - data: Dictionary containing the data. If None, load from cluster_meta file.
        - cluster_meta: Path to JSON file containing cluster metadata.
        - sampling_rate: Sampling rate for audio processing.
        - val_size: Proportion of validation data to split from the main data.
        """
        self.cluster_id = cluster_id
        self.root_path = root_path
        if data is None:
            with open(cluster_meta, mode='r') as f:
                self.data = json.load(f)
        else:
            self.data = data
        
        self.__key_type = type(list(self.data.keys())[0])
        self.sampling_rate = sampling_rate

        # Read distance metadata from CSV file and apply softmax to distances
        distance = pd.read_csv(distance_meta_data)
        self.distance_meta = {
            "speaker": distance['speaker'].to_list(),
            "p": self.__np_softmax(distance['distance_to_central'].to_list())
        }

        self.val_size = val_size
        if val_size != 0:
            self.val_data = {}
            for key, value in self.data.items():
                val_num = int(len(value) * val_size)
                self.val_data[key] = value[-val_num:]
                self.data[key] = value[:-val_num]
            self.val_idx = []
            for spk in self.distance_meta["speaker"]:
                self.val_idx += [(spk, idx) for idx in range(len(self.val_data[self.__key_type(spk)]))]
        print(f"cluster {self.cluster_id} finish init")
    def get_val_item(self, idx):
        """
        Get a validation audio item by index.
        
        Parameters:
        - idx: Index of the validation item.
        
        Returns:
        - Audio segment from the validation data.
        - speaker id
        """
        spk, val_idx = self.val_idx[idx]
        data = self.val_data[self.__key_type(spk)][val_idx]
        data_file = data['file'] if data['file'][0] != '/' else data['file'][1:]
        audio_file = os.path.join(self.root_path, data_file)
        from_idx = data['from']
        to_idx = data['to']
        audio, rate = torchaudio.load(audio_file)
        if rate != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, rate, self.sampling_rate)
        audio = audio[0, from_idx: to_idx]
        return audio, int(spk)

    def get_val_item_by_speaker(self, spk, val_idx):
        """
        Get a validation audio item by speaker and index.
        
        Parameters:
        - spk: Speaker ID.
        - val_idx: Index of the validation item.
        
        Returns:
        - Audio segment from the validation data for the given speaker.
        """
        data = self.val_data[self.__key_type(spk)][val_idx]
        data_file = data['file'] if data['file'][0] != '/' else data['file'][1:]
        audio_file = os.path.join(self.root_path, data_file)
        from_idx = data['from']
        to_idx = data['to']
        audio, rate = torchaudio.load(audio_file)
        if rate != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, rate, self.sampling_rate)
        audio = audio[0, from_idx: to_idx]
        return audio

    def get_num_speakers(self):
        """
        Get the number of speakers in the cluster.
        
        Returns:
        - Number of speakers.
        """
        return len(self.distance_meta['speaker'])

    def reset(self, num_speakers=4, logging=False, path_log="./"):
        """
        Reset the chosen speakers and their data for a new training epoch.
        
        Parameters:
        - num_speakers: Number of speakers to choose randomly.
        - logging: Whether to log the reset information.
        - path_log: Path to the log file.
        """
        self.chosen_speakers = np.random.choice(
            self.distance_meta['speaker'],
            num_speakers,
            replace=False,
            p=self.distance_meta['p']
        )
        self.chosen_data = []
        for spk in self.chosen_speakers:
            self.chosen_data += [(spk, idx) for idx in range(len(self.data[self.__key_type(spk)]))]
        if logging:
            with open(os.path.join(path_log, f"cluster_{self.cluster_id}.log"), mode='a') as f:
                f.writelines(f"reset <choose new speaker>: {self.chosen_speakers}\n")

    def __getitem__(self, idx):
        """
        Get an item from the chosen data by index.
        
        Parameters:
        - idx: Index of the item.
        
        Returns:
        - Tuple containing speaker and index.
        """
        return self.chosen_data[idx]

    def read_file(self, speaker, idx, get_ref=False):
        """
        Read an audio file segment for a given speaker and index.
        
        Parameters:
        - speaker: Speaker ID.
        - idx: Index of the segment.
        - get_ref: Whether to get a reference audio segment.
        
        Returns:
        - Audio segment, and optionally a reference audio segment.
        """
        data = self.data[self.__key_type(speaker)][idx]
        data_file = data['file'] if data['file'][0] != '/' else data['file'][1:]
        audio_file = os.path.join(self.root_path, data_file)
        from_idx = data['from']
        to_idx = data['to']
        audio, rate = torchaudio.load(audio_file)
        if rate != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, rate, self.sampling_rate)
        audio = audio[0, from_idx: to_idx]
        if not get_ref:
            return audio
        else:
            ref_idx = idx
            while ref_idx == int(idx):
                try:
                    ref_idx = int(np.random.randint(0, len(self.data[self.__key_type(speaker)])))
                except:
                    idx = 0
                    break
            data = self.data[self.__key_type(speaker)][ref_idx]
            data_file = data['file'] if data['file'][0] != '/' else data['file'][1:]
            audio_file = os.path.join(self.root_path, data_file)
            from_idx = data['from']
            to_idx = data['to']
            ref_audio, rate = torchaudio.load(audio_file)
            if rate != self.sampling_rate:
                ref_audio = torchaudio.functional.resample(ref_audio, rate, self.sampling_rate)
            ref_audio = ref_audio[0, from_idx: to_idx]
            return audio, ref_audio
        
    def get_mix_for_speaker(self, speaker_id, is_val=False):
        """
        Get a mixed audio segment for a given speaker ID.
        
        Parameters:
        - speaker_id: Speaker ID.
        - is_val: Whether to get the mixed audio from the validation set.
        
        Returns:
        - Mixed audio segment from a different speaker.
        """
        spk = int(speaker_id)
        while spk == int(speaker_id):
            spk = int(np.random.choice(self.distance_meta['speaker']))
        if not is_val:
            try:
                idx = np.random.randint(0, len(self.data[self.__key_type(spk)]))
            except:
                print(self.cluster_id,speaker_id,is_val)
                idx = 0
            return self.read_file(int(spk), idx)
        else:
            try:
                idx = np.random.randint(0, len(self.val_data[self.__key_type(spk)]))
            except:
                print(self.cluster_id,speaker_id,is_val)
                idx = 0
            return self.get_val_item_by_speaker(int(spk), idx)

    def __len__(self):
        """
        Get the length of the chosen data.
        
        Returns:
        - Length of the chosen data if available, otherwise 0.
        """
        try:
            return len(self.chosen_data)
        except:
            return 0

    def len_val_set(self):
        """
        Get the length of the validation set.
        
        Returns:
        - Length of the validation set.
        """
        return len(self.val_idx)