#!/usr/bin/env python3

import ihmm
import ihmm_io
import logging
import pickle
import time
import numpy as np
import sys
import zmq
#from scipy.sparse import *
import pyximport; pyximport.install()
from Sampler import *
import subprocess

def extractStates(index, totalK, maxes):
    (a_max, b_max, g_max) = maxes
    ## First -- we only care about a,b,g for computing next state so factor
    ## out the a,b,g
    f_ind = 0
    f_split = totalK / 2
    if index > f_split:
        index = index - f_split
        f_ind = 1
    
    j_ind = 0
    j_split = f_split / 2
    if index > j_split:
        index = index - j_split
        j_ind = 1
    
    g_ind = index % g_max
    index = (index-g_ind) / g_max
    
    b_ind = index % b_max
    index = (index-b_ind) / b_max
    
    a_ind = index % a_max
    
    ## Make sure all returned values are ints:
    return map(int, (f_ind,j_ind,a_ind, b_ind, g_ind))

def getStateIndex(f,j,a,b,g, maxes):
    (a_max, b_max, g_max) = maxes
    return (((f*2 + j)*a_max + a) * b_max + b)*g_max + g

def getStateRange(f,j,a,b, maxes):
    (a_max, b_max, g_max) = maxes
    start = getStateIndex(f,j,a,b,0,maxes)
    return range(start,start+g_max-1)


class HmmSampler(Sampler):
    
    def set_models(self, models):
        (self.models, self.pi, self.phi) = models
        
    def initialize_dynprog(self, maxLen):
        self.dyn_prog = np.zeros((get_state_size(self.models), maxLen))
        
    def forward_pass(self,dyn_prog,sent,sent_index):
        ## keep track of forward probs for this sentence:
        maxes = getVariableMaxes(self.models)
        (a_max,b_max,g_max) = maxes

        dyn_prog[:] = 0
        sentence_log_prob = 0
        
        ## Copy=False indicates that this matrix object is just a _view_ of the
        ## array -- we don't have to copy it into a matrix and recopy back to array
        ## to get the return value
        forward = np.matrix(dyn_prog, copy=False)
        for index,token in enumerate(sent):     
            ## Still use special case for 0
            if index == 0:
                g0_ind = getStateIndex(0,0,0,0,0,maxes)
                forward[g0_ind+1:g0_ind+g_max-1,0] = np.matrix(10**self.models.lex.dist[1:-1,token]).transpose()
            else:
                forward[:,index] = self.pi.transpose() * forward[:,index-1]
                forward[:,index] = np.multiply(forward[:,index], self.phi[:,token])

            normalizer = forward[:,index].sum()
            forward[:,index] /= normalizer
            
            ## Normalizer is p(y_t)
            sentence_log_prob += np.log10(normalizer)
        
        ## For the last token, multiply in the probability
        ## of transitioning to the end state. also can add up
        ## total probability of data given model here.      
        last_index = len(sent)-1
#        for state in range(0,forward.shape[0]):
#            (f,j,a,b,g) = extractStates(state, totalK, maxes)
#            forward[state,last_index] *= ((10**models.fork.dist[b,g,0] * 10**models.reduce.dist[a,1]))
                       
#            if ((last_index > 0 and (a == 0 or b == 0)) or g == 0) and forward[state, last_index] != 0:
#                logging.error("Error: Non-zero probability at g=0 in forward pass!")
#                sys.exit(-1)

        
        if np.argwhere(forward.max(0)[:,0:last_index+1] == 0).size > 0:
            logging.error("Error; There is a word with no positive probabilities for its generation")
            sys.exit(-1)

#        sentence_log_prob += np.log10( forward[:,last_index].sum() )
        
        return dyn_prog, sentence_log_prob

    def reverse_sample(self, dyn_prog, sent, sent_index):            
        sample_seq = []
        sample_log_prob = 0
        maxes = getVariableMaxes(self.models)
        totalK = get_state_size(self.models)
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        ## normalize after multiplying in the transition out probabilities
        dyn_prog[:,last_index] /= dyn_prog[:,last_index].sum()
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                        
        sample_seq.append(ihmm.State(1, extractStates(sample_t, totalK, maxes)))
        if last_index > 0 and (sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0):
            logging.error("Error: First sample has a|b|g = 0")
            sys.exit(-1)
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == 0:
                    continue

                (pf,pj,pa,pb,pg) = extractStates(ind, totalK, getVariableMaxes(self.models))
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                
                ## shouldn't be necessary, put in debugs:
                if pf == 0 and t == 1:
                    logging.warn("Found a positive probability where there shouldn't be one -- pf == 0 and t == 1")
                    dyn_prog[ind,t] = 0
                    continue
                    
                if pj == 1 and t == 1:
                    logging.warn("Found a positive probability where there shouldn't be one -- pj == 1 and t == 1")
                    dyn_prog[ind,t] = 0
                    continue
                    
                if t != 1 and pf != pj:
                    logging.warn("Found a positive probability where there shouldn't be one -- pf != pj")
                    dyn_prog[ind,t] = 0
                    continue
                
                trans_prob = 1.0
                
                if t > 0:
                    trans_prob *= 10**self.models.fork[0].dist[pb, pg, nf]

#                if nf == 0:
#                    trans_prob *= (10**self.models.reduce.dist[pa,nj])
      
                if nf == 0 and nj == 0:
                    trans_prob *= (10**self.models.act[0].dist[pa,0,na])
                elif nf == 1 and nj == 0:
                    trans_prob *= (10**self.models.root[0].dist[0,pg,na])
                elif nf == 1 and nj == 1:
                    if na != pa:
                        trans_prob = 0
      
                if nj == 0:
                    trans_prob *= (10**self.models.start[0].dist[pa,na,nb])
                else:
                    trans_prob *= (10**self.models.cont[0].dist[pb,pg,nb])
      
                trans_prob *= (10**self.models.pos.dist[nb,ng])
                              
                dyn_prog[ind,t] *= trans_prob

            dyn_prog[:,t] /= dyn_prog[:,t].sum()
            sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,t]))
            state_list = extractStates(sample_t, totalK, getVariableMaxes(self.models))
            
            sample_state = ihmm.State(1, state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
            sample_seq.append(sample_state)
            
        sample_seq.reverse()
        logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq

