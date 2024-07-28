from abc import ABC
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

class TrainingPipeline(ABC):
    def __init__(
            self,
            model:torch.nn.Module,
            train_dataset,
            val_dataset,
            optimizer = "AdamW",
            optimizer_param = {
                "lr" : 1e-3
            },
            train_batch_size = 8,
            val_batch_size = 8,
            epochs = 200,
            time_limit = 86400,
            device = None,
            using_multi_gpu = True,
            checkpoint_path = "/",
            checkpoint_name = "model.pth",
            checkpoint_rate = 1,
            patient = 3,
            checkpoint_from_epoch = 1,
            use_checkpoint = None
            ):
        super().__init__()
        self.model = model
        if use_checkpoint:
            self.model.load_state_dict(torch.load(use_checkpoint,map_location='cpu'))
        self.device = device
        self.using_multi_gpu = using_multi_gpu
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            if self.using_multi_gpu:
                if torch.cuda.device_count() == 1:
                    self.using_multi_gpu = False

        self.optimizer = getattr(optim,optimizer)(self.model.parameters(),**optimizer_param)

        if self.using_multi_gpu:
            model = torch.nn.DataParallel(model)
        
        model.to(self.device)
        self.checkpoint_file = checkpoint_path + checkpoint_name
        self.time_limit = time_limit
        self.epochs = epochs
        self.checkpoint_rate = checkpoint_rate
        self.patient = patient
        self.checkpoint_from_epoch = checkpoint_from_epoch

        self.train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,drop_last=True)
        self.val_loader = DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False)

    def checkpoint(self):
        if self.using_multi_gpu:
            torch.save(self.model.module.state_dict(),self.checkpoint_file)
        else:
            torch.save(self.model.state_dict(),self.checkpoint_file)
    
    def set_epochs(self,epochs):
        self.epochs = epochs