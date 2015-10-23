#!/usr/bin/env python3

import numpy as np

class Proposal():
    def __init__(self, ind1, ind2, sample, models, iter):
        self.ind1 = ind1
        self.ind2 = ind2
        self.backup_sample = sample
        self.models = models
        self.iter = iter

    def getAcceptanceProbability(new_sample):
        raise NotImplementedError()
        
class Merge(Proposal):
#    def __init__(self, ind1, ind2, sample):
#        Proposal.__init__(self, ind1, ind2, sample, models)

    def getAcceptanceProbability(new_sample):
        ## get prior distribution ratio:
        nc_ind1 = 0    ## Counts of index 1 in new sample
        nc_ind2 = 0    ## Counts of index 2 in new sample
        nc = 0         ## Counts of index 1 in old sample
        
        alpha = 1      ## Not sure where this comes from
        
        prior = np.log((1. / alpha) * math.factorial(nc - 1) / (math.factorial(nc_ind1 - 1) * math.factorial(nc_ind2 - 1)))
        
        ll_ratio = new_sample.log_prob - self.backup_sample.log_prob
        
        
                
class Split(Proposal):
#    def __init__(self, ind1, ind2, sample):
#        Proposal.__init__(self, ind1, ind2, sample, models)
        
    def getAcceptanceProbability(new_sample):
        ## get prior distribution ratio:
        nc_ind1 = 0    ## Counts of index 1 in new sample
        nc_ind2 = 0    ## Counts of index 2 in new sample
        nc = 0         ## Counts of index 1 in old sample
        
        alpha = 1      ## Not sure where this comes from
        
        log_prior = np.log(alpha * (math.factorial(nc_ind1 - 1) * math.factorial(nc_ind2 - 1)) / math.factorial(nc - 1))
        
        ll_ratio = new_sample.log_prob - self.backup_sample.log_prob
        
        return np.exp(ll_ratio)