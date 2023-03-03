import numpy as np
import torch
import torch.nn as nn

class EnsembleBlock(nn.Module) :

    def __init__(self,act_fn_type="relu") :
        super().__init__()
        self.layer_i  = nn.Linear(3,1000)
        self.layers_h = nn.ModuleList()
        for i in range(2) :
            self.layers_h.append( nn.Linear(1000,1000) )
        self.layer_o  = nn.Linear(1000,1)
        
        if act_fn_type == "relu" :
            self.act_fn = nn.ReLU()
        elif act_fn_type == "leaky_relu" :
            self.act_fn = nn.LeakyReLU()
        self.fin_act_fn = nn.Softplus()

    def forward(self,x) :
        x = self.layer_i(x)
        x = self.act_fn(x)
        for layer_h in self.layers_h :
            x = layer_h(x)
            x = self.act_fn(x)
        x = self.layer_o(x)
        x = self.fin_act_fn(x)
        return x
    
class EnsembleModel(nn.Module) :

    def __init__(self,block_num,act_fn_type="relu") :
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_num) :
            block = EnsembleBlock()
            self.blocks.append(block)

    def forward(self,x) :
        y = []
        for i in range(len(self.blocks)) :
            y.append( self.blocks[i](x) )
        y = torch.cat(y,axis=1)
        y = torch.mean(y,axis=1,keepdim=True)
        return y

    def __len__(self) :
        return len(self.blocks)
    
def model_load() :
    best_model_name = 'best_model_2023_2_28.pkl'
    device = 'cpu'
    block_num = 10
    model = EnsembleModel(block_num)
    model.load_state_dict(torch.load(best_model_name,map_location=device))
    return model
    
def net_infer(K,T,vol,model) :
    
    K = np.array(K)
    T = np.array(T)
    vol = np.array(vol)
    
    K_ = torch.from_numpy(K);    
    T_ = torch.from_numpy(T);    
    vol_ = torch.from_numpy(vol);
    lnK = torch.log(K_)

    x = torch.as_tensor(torch.vstack([lnK/torch.sqrt(T_),T_,vol_]).T)
    x = x.reshape([-1,3]).float()
    y = model(x)
    y = y.detach().numpy().flatten()
    return y