#!/usr/bin/env python3

import ihmm
import ihmm_io
import logging
import pickle
import time
import numpy as np
import sys
import zmq
import scipy.sparse
from PyzmqMessage import PyzmqModel
import pyximport; pyximport.install()
from Sampler import *
import subprocess

def extractStates(index, totalK, depth, maxes):
    (a_max, b_max, g_max) = maxes
    
    (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (4*depth, a_max**depth, b_max**depth, g_max))
    
    f = [-1] * depth
    j = [-1] * depth
    
    (max_d, f_val, j_val) = np.unravel_index(fj_ind, (depth, 2, 2))
    f[max_d] = f_val
    j[max_d] = j_val
    
    a = np.unravel_index(a_ind, [a_max] * depth)
    b = np.unravel_index(b_ind, [b_max] * depth)
    
    return f, j, a, b, g

def getStateIndex(f, j, a, b, g, maxes):
    (a_max, b_max, g_max) = maxes
    depth = len(f)
    for d in range(0, depth):
        if f[d] >= 0:
            cur_depth = d
            break
            
    fj_stack = np.ravel_multi_index((cur_depth, f[cur_depth], j[cur_depth]), (depth, 2, 2))
    
    a_stack  = np.ravel_multi_index(a, [a_max] * depth)
    b_stack = np.ravel_multi_index(b, [b_max] * depth)
        
    index = np.ravel_multi_index((fj_stack, a_stack, b_stack, g), (depth * 4, a_max**depth, b_max**depth, g_max))
    
    return index
    
def getStateRange(f,j,a,b, maxes):
    (a_max, b_max, g_max) = maxes
    start = getStateIndex(f,j,a,b,0,maxes)
    return range(start,start+g_max-1)

def boolean_depth(l):
    for index,val in enumerate(l):
        if val >= 0:
            return index

class HmmSampler(Sampler):
    
    def set_models(self, models):
        (self.models, self.pi) = models
        self.lexMatrix = np.matrix(10**self.models.lex.dist)
        g_len = self.models.pos.dist.shape[-1]
        w_len = self.models.lex.dist.shape[-1]
        scipy = True
        try:
            import scipy.sparse
            self.lexMultipler = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, get_state_size(self.models) / g_len)))
        except:
            logging.warn("Could not find scipy! Using numpy will be much less memory efficient!")
            self.lexMultipler = np.tile(np.identity(g_len), (1, get_state_size(self.models) / g_len))
            scipy = False
        
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
                expanded_lex = self.lexMatrix[:,token].transpose() * self.lexMultipler
                forward[:,index] = np.multiply(forward[:,index], expanded_lex.transpose())
#                forward[:,index] = np.multiply(forward[:,index], self.phi[:,token])

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
            raise ParseException("Error; There is a word with no positive probabilities for its generation")

#        sentence_log_prob += np.log10( forward[:,last_index].sum() )
        
        return dyn_prog, sentence_log_prob

    def reverse_sample(self, dyn_prog, sent, sent_index):            
        sample_seq = []
        sample_log_prob = 0
        maxes = getVariableMaxes(self.models)
        totalK = get_state_size(self.models)
        depth = len(self.models.fork.dist)
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        ## normalize after multiplying in the transition out probabilities
        dyn_prog[:,last_index] /= dyn_prog[:,last_index].sum()
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                        
        sample_seq.append(ihmm.State(1, extractStates(sample_t, totalK, depth, maxes)))
        if last_index > 0 and (sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0):
            logging.error("Error: First sample has a|b|g = 0")
            raise ParseException("Error: First sample has a|b|g = 0")
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == 0:
                    continue

                (pf,pj,pa,pb,pg) = extractStates(ind, totalK, depth, getVariableMaxes(self.models))
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                
                ## shouldn't be necessary, put in debugs:
                if pf == 0 and t == 1:
                    logging.warning("Found a positive probability where there shouldn't be one -- pf == 0 and t == 1")
                    dyn_prog[ind,t] = 0
                    continue
                    
                if pj == 1 and t == 1:
                    logging.warning("Found a positive probability where there shouldn't be one -- pj == 1 and t == 1")
                    dyn_prog[ind,t] = 0
                    continue
                    
                if t != 1 and pf != pj:
                    logging.warning("Found a positive probability where there shouldn't be one -- pf != pj")
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
            state_list = extractStates(sample_t, totalK, depth, getVariableMaxes(self.models))
            
            sample_state = ihmm.State(1, state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
            sample_seq.append(sample_state)
            
        sample_seq.reverse()
        logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq

