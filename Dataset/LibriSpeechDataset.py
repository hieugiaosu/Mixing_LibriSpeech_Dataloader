import torch 
import torch.nn as nn 
import pandas as pd
import numpy as np
import os


"""
This function is to generate the dataframe structure of librispeech dataset
@param string path: the root path of librispeech
@param bool enumerateSpeaker:  if True the speaker will be enumerate by the order
os.listdir, else it will be the id of speaker. DEFAULT True
@return DataFrame: the DataFrame structure of dataset
"""

def getLibriSpeechDataFrame(path: str,enumerateSpeaker:bool = True) -> pd.DataFrame:
    try:
        df = pd.DataFrame([],columns=['speaker','audio_file'])
        if path[-1] == '/':
            path = path[:-1]
        speakers = os.listdir(path)
        speakers = sorted(speakers,key = lambda x: int(x))
        if enumerateSpeaker:
            speakerMap = dict({int(speaker):idx for idx,speaker in enumerate(speakers)})
        else:
            # This implement is not efficient about the memmory storage but it is not 
            # affect too much, i implement like this for the easy understand code later
            speakerMap = dict({int(speaker):int(speaker) for speaker in speakers})
        for speaker in speakers:
            audioFile = list([])
            for chapter in os.listdir(f"{path}/{speaker}"):
                audiofileList = list(filter(lambda x: 'txt' not in x,os.listdir(f'{path}/{speaker}/{chapter}')))
                audiofileList = list(map(lambda x: (speakerMap[int(speaker)],f"{path}/{speaker}/{chapter}/{x}"),audiofileList))
                audioFile += audiofileList
            newRows = pd.DataFrame(audioFile,columns=['speaker','audio_file'])
            df = pd.concat([df,newRows],axis=0)
        return df
    except Exception as e:
        msg = f"Got {str(e)} in the execution\nMaybe your librispeech file structure is not correct"
        raise RuntimeError(msg)

    