import matplotlib.pyplot as plt
import ipywidgets as widgets

from bs import *
from net import *

def draw(vol,T,model) :
    
    lnK = np.linspace(-3.*vol*np.sqrt(T),3.*vol*np.sqrt(T),100)
    K = np.exp(lnK)
    true_val = bs_form(K,T,vol)
    net_val = net_infer(K,np.full_like(K,T),np.full_like(K,vol),model)
    
    plt.figure(figsize=(10,6))
    ax1 = plt.gca()
    ln1 = ax1.plot(K,true_val,':',lw=3,alpha=1,label='$c_{true}$')
    ln2 = ax1.plot(K,net_val,lw=3,alpha=0.5,label='$c_{net}$')
    ax2 = ax1.twinx()
    ln3 = ax2.plot(K,net_val-true_val,lw=1,c='C2',label='$c_{net}-c_{true}$')
    ax1.grid()
    ax1.set_xlabel('$K$')
    ax1.set_ylabel('$c$')
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc=5)
    plt.show()
    
def main() :
    
    w1 = widgets.FloatSlider(
        value=0.3,
        min=0.01,
        max=0.5,
        step=0.01,
        description='vol:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    w2 = widgets.FloatSlider(
        value=0.35,
        min=0.005,
        max=1.5,
        step=0.005,
        description='T:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
    )

    model = model_load()
    w = widgets.interact(draw,vol=w1,T=w2,model=widgets.fixed(model))