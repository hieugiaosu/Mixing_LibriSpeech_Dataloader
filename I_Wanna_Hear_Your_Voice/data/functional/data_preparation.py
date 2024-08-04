import os
import pandas as pd
import torchaudio
import re

def getLibriSpeech2MixDataFrame(
        path: str,
        enumerateSpeaker: bool = True,
        sample_rate: int = 16000,
        chunk_second_length: float = 2
    ) -> pd.DataFrame:
    try:
        count = 0
        data = []
        if path[-1] == '/':
            path = path[:-1]
        speakers = sorted(os.listdir(path), key=lambda x: int(x))
        chunk_duration = int(chunk_second_length * sample_rate)
        
        if enumerateSpeaker:
            speakerMap = {int(speaker): idx for idx, speaker in enumerate(speakers)}
        else:
            speakerMap = {int(speaker): int(speaker) for speaker in speakers}

        for speaker in speakers:
            speaker_idx = speakerMap[int(speaker)]
            speaker_path = os.path.join(path, speaker)
            
            for chapter in os.scandir(speaker_path):
                if chapter.is_dir():
                    chapter_path = chapter.path
                    audio_files = [f.path for f in os.scandir(chapter_path) if f.is_file() and not f.name.endswith('.txt')]
                    
                    for audio_file in audio_files:
                        audio_file = re.sub(r"\\","/",str(audio_file))
                        audio_info = torchaudio.info(audio_file)
                        audio_len = int(audio_info.num_frames)
                        audio_rate = audio_info.sample_rate
                        audio_len = int(audio_len*sample_rate/audio_rate)
                        count+=1
                        for i in range(0, audio_len - chunk_duration, chunk_duration):
                            data.append((speaker_idx, audio_file, i, i + chunk_duration))
        
        df = pd.DataFrame(data, columns=['speaker', 'audio_file', 'from_idx', 'to_idx'])
        mix_audio_file = []
        mix_from_idx = []
        mix_to_idx = []
        
        ref_audio_file = []
        ref_from_idx = []
        ref_to_idx = []

        change_point = [-1] + [i for i in range(len(df)-1) if df.iloc[i]['speaker'] != df.iloc[i+1]['speaker']] + [len(df)-1]
        for i in range(1,len(change_point)):
            speaker_range = change_point[i] - change_point[i-1]
            from_idx = (change_point[i]+1)%len(df)
            mix_audio_file += [df.iloc[(from_idx+offset)%len(df)]['audio_file'] for offset in range(speaker_range)]
            mix_from_idx += [df.iloc[(from_idx+offset)%len(df)]['from_idx'] for offset in range(speaker_range)]
            mix_to_idx += [df.iloc[(from_idx+offset)%len(df)]['to_idx'] for offset in range(speaker_range)]
            
        for i in range(1,len(change_point)):
            speaker_range = change_point[i] - change_point[i-1]
            from_idx = (change_point[i-1]+1)%len(df)
            ref_audio_file_i = [df.iloc[(from_idx+offset)%len(df)]['audio_file'] for offset in range(speaker_range)]
            ref_from_idx_i = [df.iloc[(from_idx+offset)%len(df)]['from_idx'] for offset in range(speaker_range)]
            ref_to_idx_i  = [df.iloc[(from_idx+offset)%len(df)]['to_idx'] for offset in range(speaker_range)]
            
            ref_audio_file += [ref_audio_file_i[(i+10)%len(ref_audio_file_i)] for i in range(len(ref_audio_file_i))]
            ref_from_idx += [ref_from_idx_i[(i+10)%len(ref_from_idx_i)] for i in range(len(ref_from_idx_i))]
            ref_to_idx += [ref_to_idx_i[(i+10)%len(ref_to_idx_i)] for i in range(len(ref_to_idx_i))]
        
        df.insert(2,"mix_audio_file",mix_audio_file,True)
        df.insert(2,"mix_from_idx",mix_from_idx,True)
        df.insert(2,"mix_to_idx",mix_to_idx,True)
        
        df.insert(2,"ref_audio_file",ref_audio_file,True)
        df.insert(2,"ref_from_idx",ref_from_idx,True)
        df.insert(2,"ref_to_idx",ref_to_idx,True)

        df['mix_audio_file'] = df['mix_audio_file'].apply(
            lambda x: re.sub(path+'/',"",x)
        )
        df['audio_file'] = df['audio_file'].apply(
            lambda x: re.sub(path+'/',"",x)
        )
        df['ref_audio_file'] = df['ref_audio_file'].apply(
            lambda x: re.sub(path+'/',"",x)
        )
        print(f'Already process {count} audio file')

        return df, count
    except Exception as e:
        msg = f"Got {str(e)} in the execution. Maybe your librispeech file structure is not correct."
        raise RuntimeError(msg)

def get_full_speaker_info_from_cluster(root_path,speaker_list = None, csv_file = None,chunk_second_length = 4, sample_rate=16000):
    count = 0
    if speaker_list is None:
        df = pd.read_csv(csv_file)
        speaker_list = df['speaker'].values.tolist()
    chunk_duration = int(chunk_second_length * sample_rate)
    data = {int(i):list([]) for i in speaker_list}
    for speaker in speaker_list:
        speaker_path = os.path.join(root_path, str(speaker))
        for chapter in os.scandir(speaker_path):
            if chapter.is_dir():
                chapter_path = chapter.path
                audio_files = [f.path for f in os.scandir(chapter_path) if f.is_file() and not f.name.endswith('.txt')]
                for audio_file in audio_files:
                    audio_file = re.sub(r"\\","/",str(audio_file))
                    audio_info = torchaudio.info(audio_file)
                    audio_len = int(audio_info.num_frames)
                    audio_rate = audio_info.sample_rate
                    audio_len = int(audio_len*sample_rate/audio_rate)
                    count+=1
                    for i in range(0, audio_len - chunk_duration, chunk_duration):
                        data[int(speaker)].append({"file":re.sub(root_path,"",audio_file),"from":i,"to":i + chunk_duration})
    print(f"process {count} files")
    return data