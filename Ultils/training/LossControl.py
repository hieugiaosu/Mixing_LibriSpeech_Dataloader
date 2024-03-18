import numpy as np
class LossLogger:
    def __init__(self,capacity = 1000,rootPath = "./") -> None:
        self.capacity = capacity
        self.loss = []
        self.minLoss = 1000
        self.path = rootPath+"logLoss.txt"
    def log(self,loss):
        if len(self.loss) >= self.capacity:
            with open(self.path,mode='a') as f: 
                f.write(','.join(list(map(str,self.loss))))
                f.write(',')
            self.loss = []
        loss = float(loss)
        self.loss.append(loss)
    def check(self):
        if len(self.loss)>100:
            loss = float(np.mean(self.loss[-100:]))
        else: 
            loss = self.loss[-1]
        if loss < self.minLoss: 
            self.minLoss = loss 
            return True 
        return False
