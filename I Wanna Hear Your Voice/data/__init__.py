from .dataset import CacheTensor, LibriSpeech2MixDataset
import pandas as pd
from sklearn.model_selection import train_test_split
KAGGLE_ROOT = "/kaggle/input/librispeech/train-clean-100/LibriSpeech/train-clean-100/"

def getDataFrameFromMetadata(path) -> pd.DataFrame:
    return pd.read_csv(path)

def getTrainAndValSetFromMetadata(
        path,
        data_source_root,
        test_size = 0.1,
        random_state = 142, 
        shuffle = True,
        sample_rate=8000,
        using_cache = True,
        cache_size = 1,
        device = 'cpu'
        ):
    df = getDataFrameFromMetadata(path)
    train_df, val_df = train_test_split(df,test_size=test_size,random_state=random_state,shuffle=shuffle)
    train_ds = LibriSpeech2MixDataset(
        train_df,
        root=data_source_root,
        sample_rate=sample_rate,
        using_cache=using_cache,
        cache_size=cache_size,
        device=device
    )

    val_ds = LibriSpeech2MixDataset(
        val_df,
        root=data_source_root,
        sample_rate=sample_rate,
        using_cache=using_cache,
        cache_size=cache_size,
        device=device
    )
    return train_ds, val_ds