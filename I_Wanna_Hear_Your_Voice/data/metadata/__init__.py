import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ClusterMetaData:
    AUDIO_FILE_META: str
    DISTANCE_META: str

current_dir = os.path.dirname(os.path.abspath(__file__))

CLUSTER_META_DATA: Dict[str, ClusterMetaData] = {}

for filename in os.listdir(current_dir):
    if filename.startswith('cluster') and filename.endswith('_meta.csv'):
        cluster_number = filename.split('_')[0].replace('cluster', '')
        
        json_filename = f'audio_file_cluster{cluster_number}.json'
        
        if os.path.exists(os.path.join(current_dir, json_filename)):

            cluster_meta = ClusterMetaData(
                AUDIO_FILE_META=os.path.join(current_dir, json_filename),
                DISTANCE_META=os.path.join(current_dir, filename)
            )

            CLUSTER_META_DATA[cluster_number] = cluster_meta

SMALL_TRAIN_CLEAN_CSV = os.path.join(current_dir, "small-train-clean.csv")

__all__ = ['CLUSTER_META_DATA','SMALL_TRAIN_CLEAN_CSV']
