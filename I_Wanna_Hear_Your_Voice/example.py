from network.models import FilterBandTFGridnet, TargetSpeakerLOTH
from data import getTrainAndValSetFromMetadata, KAGGLE_ROOT, getTrainAndValSetFromMetadataWSJ0
from training.FilterBandTFGridnetPipeline import FilterBandTFPipeline
import sys
# train_ds, val_ds = getTrainAndValSetFromMetadata(
#     "data/metadata/small-train-clean.csv",
#     "/kaggle/input/librispeech/train-clean-100/LibriSpeech/train-clean-100",
#     test_size = 0.1                                            
#     )
train_ds = getTrainAndValSetFromMetadataWSJ0(
    "data/metadata/mix_2_spk_tr.csv",
    KAGGLE_ROOT,
)
val_ds = getTrainAndValSetFromMetadataWSJ0(
    "data/metadata/mix_2_spk_cv.csv",
    KAGGLE_ROOT,
)
# model = TargetSpeakerLOTH(n_layers=5)
model = FilterBandTFGridnet(n_layers=5)


pipe = FilterBandTFPipeline(
    model = model, 
    train_dataset = train_ds, 
    val_dataset = val_ds, 
    optimizer="AdamW", 
    optimizer_param={
        "lr":1e-3,
        "weight_decay":1.0e-2
    }, 
    train_batch_size=4, 
    val_batch_size=4, 
    epochs=200, 
    time_limit=3600*12 - 360, 
    device="cuda", 
    using_multi_gpu=True, 
    checkpoint_path="./", 
    checkpoint_name="", 
    checkpoint_rate=1, 
    patient=2, 
    checkpoint_from_epoch=1, 
    use_checkpoint=None,
    warm_up = 3
)

pipe.train(40)