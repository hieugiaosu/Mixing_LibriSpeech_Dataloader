import sys 
sys.path.append('../')
sys.path.append('../../')
import model.seperate_speech.Pipeline as sep
import model.speaker_embedding.Pipeline as emb
from Ultils.data.AdjustBatch import BatchLibriMixLoader
from Ultils.training.LossControl import LossLogger
import numpy as np
import torch 
import gc
import warnings
from einops import rearrange
import time

class TrainPipeline:
    def __init__(self,e_model,e_name,e_checkPointRoot,e_lossLogRoot,e_inputConfig,
                 s_model,s_name,s_checkPointRoot, s_lossLogRoot,
                 batch_size,epoch,data,alpha = 0.2,
                 using_gpu=False,checkPointRate = 1000,state=None,timeLimit=None,
                 e_optimizer_args={},s_optimizer_args={},multi_gpu=False):
        if state is not None: np.random.seed(state)
        self.timeLimit = timeLimit
        self.alpha = alpha
        self.e_lossLogger = LossLogger(rootPath=e_lossLogRoot)
        self.s_lossLogger = LossLogger(rootPath=s_lossLogRoot)
        data.batch = 4
        self.batch_size = batch_size
        self.mixing_batch = batch_size//data.num
        self.loader = BatchLibriMixLoader(data,batch_size,data.num)
        self.e_modelControl = emb.TrainForwardPipeline(e_model,e_name,e_checkPointRoot,using_gpu,**e_optimizer_args)
        self.s_modelControl = sep.TrainForwardPipeline(s_model,s_name,s_checkPointRoot,using_gpu,**s_optimizer_args)
        self.epoch = epoch
        self.countFalse = 0
        self.checkPointRate = checkPointRate
        self.e_inputConfig = e_inputConfig
    def train(self):
        device = self.e_modelControl.getDevice()
        start_time = time.time() if self.timeLimit is not None else None
        count = 0
        stop = False
        for epoch in range(self.epoch):
            if stop: break
            for data in self.loader:
                count += 1
                e_label = data['speaker_id'].to(device)
                e_input = self.e_inputConfig(data['vocal_sample'])
                for k,v in e_input.items():
                    e_input[k] = v.to(device)
                self.e_modelControl.clear_grad()
                self.s_modelControl.clear_grad()
                s_input = {}
                s_input['x'] = data['mixing'].to(device)
                s_label = rearrange(data['audio'],"(b s) l -> b s l",b=self.mixing_batch)
                s_label = s_label.to(device)
                e_o = self.e_modelControl(e_input,e_label)
                s_input['e'] = e_o['output']

                s_o = self.s_modelControl(s_input,s_label)
                loss = s_o['loss'] + self.alpha*e_o['loss']
                loss.backward()
                self.s_modelControl.step()
                self.e_modelControl.step()

                self.e_lossLogger.log(e_o['loss'].cpu().detach().item())
                self.s_lossLogger.log(s_o['loss'].cpu().detach().item())
                print("-"*10)
                print(f'epoch: {epoch}------ iteration: {count}')
                print(f"embeding_loss: {e_o['loss'].detach().item()}\nSeprate speech loss (MSE): {s_o['loss'].detach().item()}")
                print(f"total_loss: {loss.detach().item()}")
                print("-"*10)
                del loss,e_o,s_o,e_label,s_label,e_input,s_input
                torch.cuda.empty_cache()
                gc.collect()
                if self.timeLimit:
                    spend_time = time.time() - start_time 
                    if spend_time > self.timeLimit:
                        if self.s_lossLogger.check():
                            self.e_modelControl.saveCheckPoint()
                            self.s_modelControl.saveCheckPoint()
                            self.countFalse = 0
                        else: 
                            self.countFalse += 1
                            if self.countFalse == 3:
                                stop = True 
                                warnings.warn("early stopping because out of time")
                                break
                        stop = True 
                        warnings.warn("early stopping because out of time")
                        break
                        
                if count % self.checkPointRate == 0:
                    if self.s_lossLogger.check():
                        self.e_modelControl.saveCheckPoint()
                        self.s_modelControl.saveCheckPoint()
                        self.countFalse = 0
                    else: 
                        self.countFalse += 1
                        if self.countFalse == 3:
                            stop = True 
                            warnings.warn("early stopping because loss is not decay")
                            break