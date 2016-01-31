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

def extractState(index, totalK, depth, maxes):
    (fStack, jStack, aStack, bStack, g) = extractStacks(index, totalK, depth, maxes)
    state = ihmm.State(depth)
    state.f = fStack
    state.j = jStack
    state.a = aStack
    state.b = bStack
    state.g = g
    return state
    
def extractStacks(index, totalK, depth, maxes):
    (a_max, b_max, g_max) = maxes
    
    (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (4*depth, a_max**depth, b_max**depth, g_max))
    
    f = np.zeros(depth) - 1 
    j = np.zeros(depth) - 1
    
    (max_d, f_val, j_val) = np.unravel_index(fj_ind, (depth, 2, 2))
    f[max_d] = f_val
    j[max_d] = j_val
    
    a = np.array(np.unravel_index(a_ind, [a_max] * depth))
    b = np.array(np.unravel_index(b_ind, [b_max] * depth))
    
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
        self.lexMultiplier = np.tile(np.identity(g_len), (1, get_state_size(self.models) /  g_len))
        logging.info("Lexical matrix has size %s and lex multiplier is size %s" % (self.lexMatrix.shape, self.lexMultiplier.shape))
        logging.info("Shape of the lex matrix multiplier for a single token is %s" % str(self.lexMatrix[:,1].transpose().shape))
        print("Transition matrix: \n %s", self.pi)
        
        print("Lexical matrix: \n %s", self.lexMatrix)
        print("Lexical multipler: \n %s", self.lexMultiplier)
        
    def initialize_dynprog(self, maxLen):
        self.dyn_prog = np.zeros((get_state_size(self.models), maxLen))
        
    def forward_pass(self,dyn_prog,sent,sent_index):
        try:
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
                    forward[1:g_max-1,0] = self.lexMatrix[1:-1,token]
                else:
                    forward[:,index] = self.pi.transpose() * forward[:,index-1]
                    #print("After transition %s" % forward[:,index])
                    expanded_lex = self.lexMatrix[:,token].transpose() * self.lexMultiplier  
                    #print("Expanded lex matrix: %s" % expanded_lex)
                    forward[:,index] = np.multiply(forward[:,index], expanded_lex.transpose())
                    
                normalizer = forward[:,index].sum()
                forward[:,index] /= normalizer
                #print("Matrix at index %d is %s" % (index, forward[:,index]))
                
                ## Normalizer is p(y_t)
                sentence_log_prob += np.log10(normalizer)

            last_index = len(sent)-1
            if np.argwhere(forward.max(0)[:,0:last_index+1] == 0).size > 0:
                logging.error("Error; There is a word with no positive probabilities for its generation")
                sys.exit(-1)

            ## FIXME - Do we need to multiply by -/+ at last time step for d > 0 system?
            ## TODO - should we make probability of transitioning to d > 0 state at 
            ## last time step = 0.0? Or just sample it with prob = 0.0?
        except Exception as e:
            printException()
            raise e

        return dyn_prog, sentence_log_prob

    def reverse_sample(self, dyn_prog, sent, sent_index):
        try:      
            sample_seq = []
            sample_log_prob = 0
            maxes = getVariableMaxes(self.models)
            totalK = get_state_size(self.models)
            depth = len(self.models.fork)
        
            ## Normalize and grab the sample from the forward probs at the end of the sentence
            last_index = len(sent)-1
        
            ## normalize after multiplying in the transition out probabilities
            dyn_prog[:,last_index] /= dyn_prog[:,last_index].sum()
            
            sample_depth = -1
            ## We require that the sample comes from the set of states that are at depth
            ## 0 (i.e. the sentence must fully reduce to be a valid parse)
            #print(dyn_prog[:,last_index])
            while sample_depth != 0:
                sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                sample_state = extractState(sample_t, totalK, depth, maxes)
                sample_depth = sample_state.max_awa_depth()
                #logging.debug("Sampled final state %s with depth %d" % (sample_state.str(), sample_depth))
    
            sample_seq.append(extractState(sample_t, totalK, depth, maxes))
            #logging.debug("Sampled state %s at time %d" % (sample_seq[-1].str(), last_index))
            
            if last_index > 0 and (sample_seq[-1].a[0] == 0 or sample_seq[-1].b[0] == 0 or sample_seq[-1].g == 0):
                logging.error("Error: First sample has a|b|g = 0")
                sys.exit(-1)
  
            for t in range(len(sent)-2,-1,-1):
                for ind in range(0,dyn_prog.shape[0]):
                    if dyn_prog[ind,t] == 0:
                        continue

                    prevState = extractState(ind, totalK, depth, getVariableMaxes(self.models))
                    (pf,pj,pa,pb,pg) = prevState.to_list()
                    (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                
                    ## shouldn't be necessary, put in debugs:
                    if pf[0] == 0 and t == 1:
                        logging.warn("Found a positive probability where there shouldn't be one -- pf == 0 and t == 1")
                        dyn_prog[ind,t] = 0
                        continue
                    
                    if pj[0] == 1 and t == 1:
                        logging.warn("Found a positive probability where there shouldn't be one -- pj == 1 and t == 1")
                        dyn_prog[ind,t] = 0
                        continue
                
                    prev_depth = prevState.max_awa_depth()
                    next_f_depth = sample_seq[-1].max_fork_depth()
                    next_awa_depth = sample_seq[-1].max_awa_depth()

                    if t != 0 and prev_depth == -1:
                        logging.warning("Found an empty stack at time %d" % t)
                        continue
                    
                    if prev_depth == -1:
                        prev_above_awa = 0
                    else:
                        prev_above_awa = pb[prev_depth-1]
                    
                    trans_prob = 1.0

                    ## f/j can only modify the current depth level:
                    if prev_depth != next_f_depth:
                        trans_prob = 0
                        continue
                        
                    if t > 0:
                        trans_prob *= 10**self.models.fork[next_f_depth].dist[ pb[prev_depth], pg, nf[next_f_depth] ]

                    ## Join model:
                    if nf[next_f_depth] == 0:
                        trans_prob *= (10**self.models.reduce[next_f_depth].dist[ pa[prev_depth], prev_above_awa, nj[next_f_depth] ])
                    else:
                        trans_prob *= (10**self.models.trans[next_f_depth].dist[ prev_above_awa, pg, nj[next_f_depth] ])
                        
                    ## For 4 different configurations of f and j we have these models for
                    ## active and awaited categories:
                    if nf[next_f_depth] == 0 and nj[next_f_depth] == 0:
                        trans_prob *= (10**self.models.act[next_f_depth].dist[ pa[prev_depth], prev_above_awa, na[next_f_depth] ])
                        trans_prob *= (10**self.models.start[next_f_depth].dist[ pa[prev_depth], na[next_f_depth], nb[next_f_depth] ])
                    elif nf[next_f_depth] == 1 and nj[next_f_depth] == 0:
                        trans_prob *= (10**self.models.root[next_f_depth+1].dist[pb[prev_depth], pg, na[next_f_depth+1] ])
                        trans_prob *= (10**self.models.exp[next_f_depth+1].dist[  na[next_f_depth+1], pg, nb[next_f_depth+1] ])
                    elif nf[next_f_depth] == 1 and nj[next_f_depth] == 1:
                        if na[next_f_depth] != pa[next_f_depth]:
                            trans_prob = 0
                            continue
                        trans_prob *= (10**self.models.cont[prev_depth].dist[ pb[prev_depth], pg, nb[next_f_depth] ])
                    elif nf[next_f_depth] == 0 and nj[next_f_depth] == 1:
                        if na[next_f_depth-1] != pa[next_f_depth-1]:
                            trans_prob = 0
                            continue
                        trans_prob *= (10**self.models.next[ pb[ next_f_depth-1], na[next_f_depth-1], nb[next_f_depth-1] ])

                    trans_prob *= (10**self.models.pos.dist[ nb[next_awa_depth], ng ])
                              
                    dyn_prog[ind,t] *= trans_prob

                dyn_prog[:,t] /= dyn_prog[:,t].sum()
                sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,t]))
                sample_state = extractState(sample_t, totalK, depth, getVariableMaxes(self.models))
                #logging.debug("Sampled state %s at time %d" % (sample_state.str(), t))
            
                if t > 0 and sample_state.g == 0:
                    logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
                sample_seq.append(sample_state)
            
            sample_seq.reverse()
            logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        except Exception as e:
            printException()
            raise e
            
        return sample_seq

