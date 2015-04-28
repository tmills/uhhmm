#!#!/usr/bin/env python3.4

import numpy as np
import logging

## TODO -- instead of being passed H have the Model class implement
## a get_base_distribution() method that can be called
def sampleDirichlet(counts, H, nullState = False):
    L = max(H.shape)
    K = counts.pairCounts.shape[0]
    P = np.zeros(counts.pairCounts.shape)
    
#    logging.debug("Shape of P is %s", P.shape)
    
    ## Sample from a dirichlet to get distribution over
    ## words for every pos tag:
    for i in range(0,K):
        base = counts.pairCounts[i,:]+H
#        logging.debug("i is %d", i)
#        logging.debug("Shape of pairCounts slice is %s", counts.pairCounts[i,:].shape)
#        logging.debug("Shape of H is %s", H.shape)
#        logging.debug("Shape of base is %s", base.shape)
#        logging.debug("Shape of P is %s", P.shape)
#        logging.debug("Shape of P[i,:] is %s", P[i,:].shape)
#        logging.debug(base)
        ## base[0][0] will be 0 but needs to be positive for gamma
        if nullState:
            base[0][0] = 1
        P[i,:] = np.random.gamma(base,1)
        if nullState:
            P[i,0] = 0
        P[i,:] /= sum(P[i,:])

    return P