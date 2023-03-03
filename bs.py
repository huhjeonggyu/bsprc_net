import numpy as np
from scipy.stats import norm

def bs_form(K,T,vol) :
    d1 = (np.log(1./K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return norm.cdf(d1) - K*norm.cdf(d2)