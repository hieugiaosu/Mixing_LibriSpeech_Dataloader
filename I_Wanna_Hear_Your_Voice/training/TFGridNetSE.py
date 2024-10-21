from torch.nn.modules import Module
from training.utils import TrainingPipeline
from criterion import SingleSrcNegSDRScaledEst,Mixture_constraint_loss

from torch.cuda.amp import GradScaler, autocast
import torch
import time
import gc
from torch.utils.data import DataLoader

class TFGridNetSEPipeLine(TrainingPipeline):
    def __init__(
            self, 
            model: Module, 
            train_dataset, 
            val_dataset, 
            optimizer="AdamW", 
            optimizer_param={
                "lr":1e-3,
                "weight_decay":1.0e-2
            }, 
            train_batch_size=8, 
            val_batch_size=8, 
            epochs=200, 
            time_limit=86400, 
            device=None, 
            using_multi_gpu=True, 
            checkpoint_path="/", 
            checkpoint_name="model.pth", 
            checkpoint_rate=1, 
            patient=3, 
            checkpoint_from_epoch=1, 
            use_checkpoint=None,
            train_dataloader_class = DataLoader,
            val_dataloader_class = DataLoader,
            warm_up = 3,
            checkpoint_call_back = None
            ):
        super().__init__(model, train_dataset, val_dataset, optimizer, optimizer_param, train_batch_size, val_batch_size, epochs, time_limit, device, using_multi_gpu, checkpoint_path, checkpoint_name, checkpoint_rate, patient, checkpoint_from_epoch, use_checkpoint, train_dataloader_class, val_dataloader_class,checkpoint_call_back)
        print("This pipeline is train in mixed precision")
        self.si_sdr_fn = SingleSrcNegSDRScaledEst(reduction="mean")
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.mixture_constraint_fn = Mixture_constraint_loss()
        self.scaler = GradScaler()
        self.warm_up = warm_up

    def train_iter(self,epoch,start_time):
        print(f"----------------------------train---------epoch: {epoch}/{self.epochs}--------------------------")
        if self.using_multi_gpu:
            self.model.module.train()
        else:
            self.model.train()
        
        tot_loss, num_batch = 0, 0
        total_batch = len(self.train_loader)

        for data in self.train_loader:
            self.optimizer.zero_grad()
            mix = data['mix'].to(self.device)
            src0 = data['src0'].to(self.device)
            auxs = data['auxs'].to(self.device)
            speaker_id = data['speaker_id'].to(self.device)
            num_batch += 1
            with autocast():  # Use autocast for mixed precision
                yHat, speakers_pred = self.model(mix, auxs)
                si_sdr_loss = self.si_sdr_fn(yHat,src0)
                ce_loss = self.ce_loss(speakers_pred, speaker_id)
                loss = si_sdr_loss + 0.5*ce_loss
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.module.parameters() if self.using_multi_gpu else self.model.parameters(), 
                5.0, norm_type=2.0, error_if_nonfinite=False, foreach=None
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            tot_loss += loss.cpu().detach().item()
            print(f"--------------batch:{num_batch}/{total_batch}---------loss:{loss.cpu().detach().item()}|si-sdr:{si_sdr_loss.cpu().detach().item()}----------|ce-loss: {ce_loss.cpu().detach().item()}")
            del mix, src0, yHat, loss, si_sdr_loss, ce_loss, auxs, speaker_id
            torch.cuda.empty_cache()
            gc.collect()
            if time.time() - start_time > self.time_limit:
                print('-------------------out of time-----------------------')
                break
        return tot_loss / num_batch, num_batch
    
    def validate_iter(self):
        print("-------------------------validate---------------------------")
        if self.using_multi_gpu:
            self.model.module.eval()
        else:
            self.model.eval()
        tot_loss, num_batch = 0, 0
        with torch.no_grad():
            for data in self.val_loader:
                mix = data['mix'].to(self.device)
                src0 = data['src0'].to(self.device)
                auxs = data['auxs'].to(self.device)
                speaker_id = data['speaker_id'].to(self.device)
                num_batch += 1
                with autocast():  # Use autocast for mixed precision
                    yHat, speakers_pred = self.model(mix, auxs)
                    si_sdr_loss = self.si_sdr_fn(yHat,src0)
                    ce_loss = self.ce_loss(speakers_pred, speaker_id)
                    loss = si_sdr_loss + 0.5*ce_loss
                tot_loss += loss.cpu().detach().item()
                del mix, src0, yHat, loss, si_sdr_loss, ce_loss, auxs, speaker_id
                torch.cuda.empty_cache()
                gc.collect()
        return tot_loss / num_batch, num_batch
    
    def train(self,initial_loss = 40):
        best_loss = initial_loss
        count = 0
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            train_start_time = time.time()
            train_loss, train_num_batch = self.train_iter(epoch,start_time)
            train_end_time = time.time()
            print(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss = {train_loss:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
            if epoch % self.checkpoint_rate == 0 and epoch >= self.checkpoint_from_epoch:
                valid_start_time = time.time()
                val_loss, valid_num_batch = self.validate_iter()
                valid_end_time = time.time()
                print(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss (SI-SDR) = {val_loss:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")
                if val_loss < best_loss:
                    self.checkpoint()
                    count = 0
                    best_loss = val_loss
                else:
                    count += 1
                    if count > self.patient:
                        print('early stopping because loss is not decreasing')
                        break
            if time.time() - start_time > self.time_limit:
                print("-------------out of time------------------")
                break