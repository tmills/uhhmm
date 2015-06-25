#!#!/usr/bin/env python3.4

import numpy as np
import logging

## TODO -- instead of being passed H have the Model class implement
## a get_base_distribution() method that can be called
def sampleDirichlet(counts, H):
    L = max(H.shape)
    K = counts.pairCounts.shape[0]
    P = np.zeros(counts.pairCounts.shape)
    
    ## Sample from a dirichlet to get distribution over
    ## words for every pos tag:
    for i in range(0,K):
        ## Set last value (new table probability):
        if H.ndim == 2:
            hrow = H[0]
        elif H.ndim == 1:
            hrow = H

        P[i,-1] = hrow[-1]
                
        ## Get distribution for existing table by slicing out the null state
        ## and the new table state:
        base = np.zeros((1,L-2)) + counts.pairCounts[i,1:-1]+hrow[1:-1]
        
        if base.shape[1] == 1:
            P[i,1] = 1.0
        else:
            ## Get the distribution normalized relative to the new table prob:
            dist = np.random.gamma(base, 1)
            dist /= dist.sum()
            dist *= (1 - P[i,-1])
            P[i,0] = -np.inf
            P[i,1:-1] = np.log10(dist)
            P[i,-1] = np.log10(P[i,-1])

    return P

def sampleBernoulli(counts, H):
    ## How many inputs?
    K = counts.pairCounts.shape[0]
    assert counts.pairCounts.shape[1] == 2, 'This method can only be called for a distribution with 2 outputs (F or J models)'
    
    P = np.zeros((K,2))
    
    for i in range(0,K):
        base = counts.pairCounts[i,:] + H.flatten()
        P[i,:] = np.log10(np.random.dirichlet(base, 1)[0])
        
    return P
        