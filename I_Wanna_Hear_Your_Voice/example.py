from network.models import FilterBandTFGridnet
from data import getTrainAndValSetFromMetadata, KAGGLE_ROOT
from training.FilterBandTFGridnetPipeline import FilterBandTFPipeline

train_ds, val_ds = getTrainAndValSetFromMetadata(
    "data/metadata/small-train-clean.csv",
    "/kaggle/input/librispeech/train-clean-100/LibriSpeech/train-clean-100",
    test_size = 0.1                                            
    )

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
    train_batch_size=8, 
    val_batch_size=8, 
    epochs=200, 
    time_limit=3600*12 - 360, 
    device='cuda', 
    using_multi_gpu=True, 
    checkpoint_path="./", 
    checkpoint_name="FilterBandTFGridnet.pth", 
    checkpoint_rate=1, 
    patient=2, 
    checkpoint_from_epoch=1, 
    use_checkpoint=None,
    warm_up = 3
)

pipe.train(40)