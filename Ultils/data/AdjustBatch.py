import torch

class BatchLibriMixLoader:
    def __init__(self,loader,batch_size,num_mixing):
        assert batch_size % num_mixing == 0, "batch and num mixing is not compatible"
        self.loader = loader
        self.batch_size = batch_size
        self.num_mixing = num_mixing
    def __iter__(self):
        self.iterLoader = iter(self.loader)
        return self
    def __next__(self):
        try: 
            self.data = next(self.iterLoader)
            while self.data['audio'].shape[0] < self.batch_size:
                data = next(self.iterLoader)
                for k,v in self.data.items():
                    self.data[k] = torch.cat([v,data[k]],dim=0)
            for k,v in self.data.items():
                if k == 'mixing':
                    self.data[k] = v[:self.batch_size//self.num_mixing,:]
                elif k == 'speaker_id':
                    self.data[k] = v[:self.batch_size]
                else:
                    self.data[k] = v[:self.batch_size,:]
            return self.data
        except StopIteration:
            raise StopIteration()