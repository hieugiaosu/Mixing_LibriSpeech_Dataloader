from model.seperate_speech import Module
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import sys 
sys.path.append('../')
sys.path.append('../../')

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
        self.lossfn = Module.SI_SDRLoss(2)
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
        o = self.model(**data).squeeze()
        loss = self.lossfn(o,label).mean()
        return {"output":o,"loss":loss}
    
    def step(self):
        self.optimizer.step()

    def getDevice(self):
        return self.device