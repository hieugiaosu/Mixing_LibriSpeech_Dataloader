import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchaudio.transforms as T 
from einops import rearrange,repeat

class DualBlock(nn.Module):

    ## Input for this block will have the shape (B,S,L)
    def __init__(self,embeding_dim,hidden_k,window_size,stack=4 ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if embeding_dim % 2 == 0:
            self.padding = embeding_dim//2
            self.crop = True
        else: 
            self.padding = (embeding_dim-1)//2
            self.crop = False
        self.hidden_k = hidden_k
        self.window_size = window_size
        self.stack = stack
        self.emb_i_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
        self.emb_o_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
        self.positiveLine = nn.ModuleList([
            nn.Conv1d(1,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same")
        ])
        self.negativeLine = nn.ModuleList([
            nn.Conv1d(1,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same"),
            nn.Conv1d(hidden_k,hidden_k,window_size,padding="same")
        ])
        self.bnormP = nn.BatchNorm1d(hidden_k)
        self.bnormN = nn.BatchNorm1d(hidden_k)
    def forward(self,x,e,reshape_input = True,reshape_output=True,batch = None):
        if (not reshape_input) and reshape_output:
            assert batch is not None, "we need batch to reshape"
        if reshape_input:
            batch = x.shape[0]
            i = rearrange(x,"b s l -> (b s) l")
        else: 
            i = x 
        i = i.unsqueeze(0)
        ## i now (1,b*s,l)
        e_i = self.emb_i_trans(e.unsqueeze(1))
        ### now e_i will have shape (B*S,1,E)
        i = F.conv1d(i,e_i,padding = self.padding,groups=i.shape[1]).squeeze()
        i = i.unsqueeze(1)
        if self.crop:
            i = i[:,:,:-1]
        i = F.tanh(i)
        ## now i will have shape (B*S,1,L) or (B*S,1,L+1)
        
        p_i = F.relu(i)
        n_i = -F.relu(-i)
        ## positive flow:
        p_1 = F.tanh(self.positiveLine[0](p_i))
        p_2 = F.tanh(self.positiveLine[1](p_1))
        p_3 = F.tanh(self.positiveLine[2](p_2+p_i))
        p_o = self.bnormP(p_3+p_1)
        ## negative flow:
        n_1 = F.tanh(self.negativeLine[0](n_i))
        n_2 = F.tanh(self.negativeLine[1](n_1))
        n_3 = F.tanh(self.negativeLine[2](n_2+n_i))
        n_o = self.bnormN(n_3+n_1)

        o = n_o + p_o 
        ##o have shape (B*S,hidden_k,L)
        o = o.mean(dim=1)
        o = o.unsqueeze(0)
        e_o = self.emb_o_trans(e.unsqueeze(1))
        o = F.conv1d(o,e_o,padding = self.padding,groups=o.shape[1]).squeeze()
        o = o.unsqueeze(1)
        if self.crop:
            o = o[:,:,:-1]
        o = F.tanh(o)
        
        o = o.squeeze()
        if reshape_output:
            o = rearrange(o,"(b s) l -> b s l",b=batch)
        return o

class DualBlockBaseModel(nn.Module):
    def __init__(self,depth ,embeding_dim,hidden_k,window_size,stack=4,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.ModuleList([DualBlock(embeding_dim,hidden_k,window_size,stack)]*depth)
        if embeding_dim % 2 == 0:
            self.padding = embeding_dim//2
            self.crop = True
        else: 
            self.padding = (embeding_dim-1)//2
            self.crop = False
        self.emb_i_trans = nn.Sequential(
            nn.Conv1d(1,stack,1),
            nn.Tanh(),
            nn.Conv1d(stack,1,1),
            nn.Softmax(dim=-1)
        )
    def forward(self,x,e):
        ## x is in shape (B,L), e in shape (B*S,E)
        batch = x.shape[0]
        i = x.unsqueeze(0)
        e_i = self.emb_i_trans(e.unsqueeze(1))

        i = F.conv1d(i,e_i,padding = self.padding,groups=batch)
        if self.crop:
            i = i[:,:,:-1]
        ## now i has shape (1,B*S,L)
        l_i = i.squeeze()
        l_o = 0
        for idx in range(len(self.block)):
            l_i = F.tanh(l_i + l_o)
            l_o = self.block[idx](l_i,e,reshape_input = False,reshape_output = (idx == len(self.block)-1),batch=batch)
        return l_o

class SI_SDRLoss(nn.Module):
    '''num batch dim is how many dim in tensor is a batch. 
    for example input can be (B,L) with batch is batch size and L is audio Length
    or it can be: (B,S,L) with batch is batch size, S is number of speaker in each
    batch and L is audio Length.
    For this version, I only support those kind of shape. So numBatchDim is only
    2 or 3
    '''
    def __init__(self,numBatchDim:int=2 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert numBatchDim == 2 or numBatchDim == 3, "only support 2 or 3"
        self.numBatchDim = numBatchDim
    def forward(self,input,label):
        assert input.shape == label.shape, "2 input of loss function must have the same shape"
        assert input.dim() == self.numBatchDim, f"numBatchDim is set to {self.numBatchDim} but get {input.shape} tensor as a input"
        if self.numBatchDim == 3:
            ## now is (B,S,L)
            input = rearrange(input,"b s l -> (b s) l")
            label = rearrange(label,"b s l -> (b s) l")
        term1 = torch.bmm(input.unsqueeze(1),label.unsqueeze(2)).squeeze()
        term2 = torch.bmm(label.unsqueeze(1),label.unsqueeze(2)).squeeze()
        alpha = term1/(term2+1e-6) 
        term3 = alpha.unsqueeze(1)*label-input
        term4 = torch.bmm(term3.unsqueeze(1),term3.unsqueeze(2)).squeeze() + 1e-6
        loss = -10*torch.log10(((alpha**2)*term2 + 1e-6)/term4)
        return loss.mean(0)

class EfficientAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)    
    def forward(self,q,k,v):
        q_n = F.softmax(q,dim=-1)
        k_n = F.softmax(k,dim=-1)
        k_n = k_n.transpose(1,2)
        return torch.bmm(q_n,torch.bmm(k_n,v))

class Conv1dBlock(nn.Module):
    def __init__(self,inChannel,hiddenChannel,outChannel,windowSize,padding="same",stride=1,dilation=1, groups=1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv1d(inChannel,hiddenChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      ),
            nn.ELU(),
            nn.Conv1d(hiddenChannel,hiddenChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      ),
            nn.ELU(), 
            nn.Conv1d(hiddenChannel,outChannel,windowSize,stride=stride,padding=padding,
                      groups=groups,dilation=dilation
                      )
        )
    def forward(self,x):
        return self.block(x)
class UnitBlock(nn.Module):
    def __init__(self,inChannel, outChannel,inputLength ,embHiddenChannel=171,embedingDim=512 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.audioTransform = Conv1dBlock(
            inChannel,
            (outChannel+inChannel)//2,
            outChannel,inputLength//100 + 1,
            )
        self.embTransform = Conv1dBlock(
            embHiddenChannel,
            (outChannel+embHiddenChannel)//2,
            outChannel,1
        )
        self.qExtract1 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,inputLength//100,stride=inputLength//100),
            nn.ELU(),
            nn.Linear(100,embedingDim)
        )
        self.vExtract1 = nn.Conv1d(outChannel,outChannel,1)
        self.crossAttention1 = EfficientAttention()
        self.layerNorm = nn.LayerNorm((outChannel,inputLength))
        self.convLayer = Conv1dBlock(outChannel,outChannel,outChannel,inputLength//100 + 1)
        self.qExtract2 = nn.Sequential(
            nn.Conv1d(outChannel,outChannel,inputLength//100,stride=inputLength//100),
            nn.ELU(),
            nn.Linear(100,embedingDim)
        )
        self.vExtract2 = nn.Conv1d(outChannel,outChannel,1)
        self.crossAttention2 = EfficientAttention()
    def forward(self,audio,emb_hidden):
        audio_i = self.audioTransform(audio)
        emb_i = self.embTransform(emb_hidden)
        q1 = self.qExtract1(audio_i)
        v1 = self.vExtract1(audio_i)
        att = self.crossAttention1(q=q1,k=emb_i,v=v1)
        o = self.convLayer(att)
        o = self.layerNorm(att+o)
        q2 = self.qExtract1(o)
        v2 = self.vExtract1(o)
        output = self.crossAttention2(q=q2,k=emb_i,v=v2)
        return output
class FiLMBlock(nn.Module):
    def __init__(self,inDim, outFiLMFeatures,transform ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(inDim,outFiLMFeatures*2),
            nn.ELU(),
            nn.Linear(outFiLMFeatures*2,outFiLMFeatures*2)
        )
        self.filmFeature = outFiLMFeatures
        self.transform = transform
    def forward(self,condition,*args,**kwwargs):
        film =self.model(condition)
        gamma = film[:,:self.filmFeature]
        beta = film[:,self.filmFeature:]
        y = self.transform(*args,**kwwargs)
        return gamma[:,:,None]*y+beta[:,:,None]
class AEBaseModel(nn.Module):
    def __init__(self,inputLength ,embHiddenChannel=171,embedingDim=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left = nn.ModuleList([
            UnitBlock(1,128,inputLength,embHiddenChannel,embedingDim),
            UnitBlock(128,256,inputLength//4,embHiddenChannel,embedingDim),
            UnitBlock(256,512,inputLength//16,embHiddenChannel,embedingDim)
        ])
        self.right = nn.ModuleList([
            UnitBlock(512,256,inputLength//16,embHiddenChannel,embedingDim),
            UnitBlock(256,128,inputLength//4,embHiddenChannel,embedingDim),
            UnitBlock(128,128,inputLength,embHiddenChannel,embedingDim)
        ])
        self.beforLast = FiLMBlock(embedingDim,128,nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.ELU()
        ))
        self.lastLayer = nn.Sequential(
            nn.Conv1d(128,128,inputLength//100+1,padding="same"),
            nn.ELU(),
            nn.Conv1d(128,1,inputLength//100+1,padding="same"),
            nn.Tanh()
        )
        self.downSample = nn.ModuleList([
            FiLMBlock(embedingDim,128,nn.AvgPool1d(4)),
            FiLMBlock(embedingDim,256,nn.AvgPool1d(4))
            ])
        self.middle = FiLMBlock(embedingDim,512,nn.Conv1d(512,512,1))
        self.upSample = nn.ModuleList([
            FiLMBlock(embedingDim,256,nn.ConvTranspose1d(256,256,4,stride=4)),
            FiLMBlock(embedingDim,128,nn.ConvTranspose1d(128,128,4,stride=4))
            ])
        self.norm = nn.ModuleList([
            nn.LayerNorm([256,inputLength//16]),
            nn.LayerNorm([256,inputLength//4]),
            nn.LayerNorm([128,inputLength]),
            ])
    def forward(self,audio,emb_hidden,emb):
        l1o = self.left[0](audio,emb_hidden)
        l2i = self.downSample[0](emb,l1o)
        l2o = self.left[1](l2i,emb_hidden)
        l3i = self.downSample[1](emb,l2o)
        l3o = self.left[2](l3i,emb_hidden)
        l4i = self.middle(emb,l3o)
        l4o = self.right[0](l4i,emb_hidden)
        l4o = self.norm[0](l4o+l3i)
        l5i = self.upSample[0](emb,l4o,output_size=l2o.size())
        l5i = self.norm[1](l5i+l2o)
        l5o = self.right[1](l5i,emb_hidden)
        l6i = self.upSample[1](emb,l5o,output_size=l1o.size())
        l6i = self.norm[2](l6i+l1o)
        l6o = self.right[2](l6i,emb_hidden)
        o = self.beforLast(emb,l6o)
        o = self.lastLayer(o) 
        return o
    
class AEInputConfigAfterEmbedding:
    def __init__(self) -> None:
        pass
    def __call__(self,e_output,audio):
        return {"audio": repeat(audio['mixing'],"b l -> (b r) 1 l", r=audio["audio"].shape[0]//audio["mixing"].shape[0]),
                "emb_hidden": e_output['last_hidden'], "emb": e_output['output']
                }