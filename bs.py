import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn

def bs_form(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return norm.cdf(d1) - K*norm.cdf(d2)

def bs_dual_delta(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return -norm.cdf(d2)

def bs_dual_gamma(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return norm.pdf(d2)/(K*vol*np.sqrt(T))

def bs_theta(K,T,vol) :
    return -0.5*bs_vega(K,T,vol)*vol/T

def bs_vega(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    return norm.pdf(d1)*np.sqrt(T)

def bs_vomma(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return norm.pdf(d1)*np.sqrt(T)*d1*d2/vol

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
    # x : [log(K),T,vol]
    # y : c
    model = EnsembleModel(block_num)
    model.load_state_dict(torch.load(best_model_name,map_location=device))
    return model
    
def deriv(y,x) :
    dx1_y = torch.autograd.grad(y,x,create_graph=True,
                            grad_outputs = torch.ones_like(y),
                            allow_unused = True,
                            retain_graph = True)[0]
    dx2_y = torch.autograd.grad(dx1_y,x,create_graph=True,
                            grad_outputs = torch.ones_like(dx1_y),
                            allow_unused = True,
                            retain_graph = True)[0]
    dx1_y = dx1_y.detach().numpy()
    dx2_y = dx2_y.detach().numpy()
    return (dx1_y,dx2_y)

def infer(K,T,vol,model) :
    
    K = np.array(K)
    T = np.array(T)
    vol = np.array(vol)
    
    oshape = K.shape
    
    K_ = torch.from_numpy(K);    
    T_ = torch.from_numpy(T);    
    vol_ = torch.from_numpy(vol);
    lnK = torch.log(K_)

    x = torch.as_tensor(torch.vstack([lnK/torch.sqrt(T_),T_,vol_]).T)
    x = x.reshape([-1,3]).float()
    y = model(x)
    y = y.detach().numpy().flatten()
    
    return y

def infer2(K,T,vol,model) :
    
    oshape = K.shape
    
    K_ = torch.from_numpy(K);    K_.requires_grad = True
    T_ = torch.from_numpy(T);    T_.requires_grad = True
    vol_ = torch.from_numpy(vol);vol_.requires_grad = True
    lnK = torch.log(K_)

    x = torch.as_tensor(torch.vstack([lnK/torch.sqrt(T_),T_,vol_]).T)
    x = x.reshape([-1,3]).float()
    y = model(x)
    dK1_y,dK2_y = deriv(y,K_)
    dT1_y,_ = deriv(y,T_)
    dvol1_y,dvol2_y = deriv(y,vol_)
    y = y.detach().numpy().flatten()
    
    return (y,dK1_y,dK2_y,dT1_y,dvol1_y,dvol2_y)

def true_val(K,T,vol) :
    y_true = bs_form(K,T,vol) 
    dK1_y_true = bs_dual_delta(K,T,vol)
    dK2_y_true = bs_dual_gamma(K,T,vol)
    dt1_y_true = bs_theta(K,T,vol)
    dvol1_y_true = bs_vega(K,T,vol)
    dvol2_y_true = bs_vomma(K,T,vol)
    return (y_true,dK1_y_true,dK2_y_true,dt1_y_true,dvol1_y_true,dvol2_y_true)   