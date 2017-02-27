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
        ## Get distribution for existing table by slicing out the null state
        ## and the new table state:
        base = np.zeros((1, counts.shape[-1]))
        if counts.shape == H.shape:
            ## Case for hierarchical prior:
            #logging.info("Ind is %s, shape of counts is %s, shape of H is %s" % ( str(ind), counts[ind][1:].shape, H[ind][1:].shape) )
            base += counts[ind][:] + H[ind][:]
        elif len(H.shape) == 2:
            ## Case where the vector is a 2-d matrix with one dim of size 1
            base += counts[ind][:] + H[0,:]
        elif len(H.shape) == 1:
            ## Case where the vector is just an array
            base += counts[ind][:] + H[:]
        else:
            logging.warning("Could not compute base with counts shape %s and base shape %s" % (counts.shape, H.shape) )
            
        if base.shape[1] == 1:
            P[ind,1] = 1.0
        elif base[0,0] > 0:
            P[ind][:] = np.log10(sampleSimpleDirichlet(base))
        else:
            P[ind][0] = -np.inf
            P[ind][1:] = np.log10(sampleSimpleDirichlet(base[:,1:]))

    return P

def sampleSimpleDirichlet(base):
    dist = np.random.gamma(base, 1)
    dist /= dist.sum()
    return dist
    
def sampleBernoulli(counts, H):
    ## How many inputs?
    K = counts.shape[0:-1]
    assert counts.shape[-1] == 2, 'This method can only be called for a distribution with 2 outputs (F or J models)'
    
    P = np.zeros(counts.shape)
    
    for ind,val in np.ndenumerate(counts[...,0]):
        if counts.shape == H.shape:
            base = counts[ind][:] + H[ind][:]
        else:
            base = counts[ind][:] + H.flatten()
        P[ind][:] = np.log10(sampleSimpleBernoulli(base))
        
    return P
        
        
def sampleSimpleBernoulli(base):
    return np.random.dirichlet(base)
    
