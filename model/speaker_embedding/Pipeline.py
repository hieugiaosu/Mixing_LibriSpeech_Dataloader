from model.speaker_embedding import Module
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import sys 
import gc
import numpy as np
sys.path.append('../')
sys.path.append('../../')
from Ultils.data.AdjustBatch import BatchLibriMixLoader
from Ultils.training.LossControl import LossLogger

class TrainForwardPipeline:
    def __init__(self,model,name,checkPointRoot,using_gpu=False,default_device ='cpu',**optimizer_args):
        self.model = model 
        self.checkPointRoot = checkPointRoot
        if checkPointRoot[-1] != '/': self.checkPointRoot+='/'
        self.optimizer = optim.AdamW(self.model.parameters(),**optimizer_args)
        self.name = name
        self.device = default_device
        self.multi = False
        if using_gpu:
            if not torch.cuda.is_available():
                warnings.warn("using gpu is on but there aren't any available cuda so the model will run on cpu")
            # elif torch.cuda.device_count() >= 2:
            #     self.model = nn.DataParallel(self.model)
            #     self.model.to('cuda')
            #     self.device = 'cuda'
            #     self.multi = True
            else:
                if self.device == 'cpu':
                    self.device = 'cuda'
                self.model.to(self.device)
                
        self.model.train()
        self.lossfn = Module.EmbeddingLoss()
    def saveCheckPoint(self):
        torch.save(self.optimizer.state_dict(),f"{self.checkPointRoot}{self.name}_optimizer.pth")
        torch.save(self.model.state_dict() if not self.multi else self.model.module.state_dict(),
                   f"{self.checkPointRoot}{self.name}_model.pth"
                   )
    def loadOptimCheckPoint(self,path):
        self.optimizer.load_state_dict(torch.load(path))
    
    def clear_grad(self):
        self.optimizer.zero_grad()

    def __call__(self,data:dict,label):
        o = self.model(**data)
        loss = self.lossfn(o,label).mean()
        return {"output":o,"loss":loss}
    
    def step(self):
        self.optimizer.step()

    def getDevice(self):
        return self.device

class TrainPipeline:
    def __init__(self,model,name,checkPointRoot,
                 batch_size,epoch,data,lossLogRoot,inputConfig,
                 using_gpu=False,checkPointRate = 1000,state=None,
                 **optimizer_args):
        if state is not None: np.random.seed(state)
        self.lossLogger = LossLogger(rootPath=lossLogRoot)
        data.batch = 8
        self.loader = BatchLibriMixLoader(data,batch_size,data.num)
        self.modelControl = TrainForwardPipeline(model,name,checkPointRoot,using_gpu,**optimizer_args)
        self.epoch = epoch
        self.countFalse = 0
        self.checkPointRate = checkPointRate
        self.inputConfig = inputConfig
    def train(self):
        device = self.modelControl.getDevice()
        count = 0
        stop = False
        for epoch in range(self.epoch):
            if stop: break
            for data in self.loader:
                count += 1
                label = data['speaker_id'].to(device)
                input = self.inputConfig(data['vocal_sample'])
                for k,v in input.items():
                    input[k] = v.to(device)
                o = self.modelControl(input,label)
                o['loss'].backward()
                self.modelControl.step()
                self.lossLogger.log(o['loss'].cpu().detach().item())
                print(f"\r---epoch:{epoch}----loss: {o['loss'].detach().item()}----------",end="",flush=True)
                del o, input,label 
                torch.cuda.empty_cache()
                gc.collect()
                if count % self.checkPointRate == 0:
                    if self.lossLogger.check():
                        self.modelControl.saveCheckPoint()
                        self.countFalse = 0
                    else: 
                        self.countFalse += 1
                        if self.countFalse == 3:
                            stop = True 
                            warnings.warn("early stopping because loss is not decay")
                            break
