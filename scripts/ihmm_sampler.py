#!#!/usr/bin/env python3.4

import numpy as np
import logging
import sys

## TODO -- instead of being passed H have the Model class implement
## a get_base_distribution() method that can be called
def sampleDirichlet(counts, H):
    L = max(H.shape)
    K = counts.shape[0:-1]
    P = np.zeros(counts.shape)
    
    ## Sample from a dirichlet to get distribution over
    ## words for every pos tag:
    
    for ind,val in np.ndenumerate(counts[...,0]):
        ## Set last value (new table probability):
        
        H = H.flatten()
                
        ## Get distribution for existing table by slicing out the null state
        ## and the new table state:
        base = np.zeros((1,L-1)) + counts[ind][1:]+H[1:]
        
        if base.shape[1] == 1:
            P[ind,1] = 1.0
        else:
            P[ind][0] = -np.inf
            P[ind][1:] = np.log10(sampleSimpleDirichlet(base))

    return P

def sampleSimpleDirichlet(base):
    if (base.ndim==1 and base.shape[0] == 1) or (base.ndim==2 and base.shape[1] == 1):
        return np.array([1.0])

    dist = np.random.gamma(base, 1)
    dist /= dist.sum()
    return dist
    
def sampleBernoulli(counts, H):
    ## How many inputs?
    K = counts.shape[0:-1]
    assert counts.shape[-1] == 2, 'This method can only be called for a distribution with 2 outputs (F or J models)'
    
    P = np.zeros(counts.shape)
    
    for ind,val in np.ndenumerate(counts[...,0]):
        base = counts[ind][:] + H.flatten()
        P[ind][:] = np.log10(sampleSimpleBernoulli(base))
        
    return P
        
        
def sampleSimpleBernoulli(base):
    return np.random.dirichlet(base)
    