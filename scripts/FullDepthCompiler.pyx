#!/usr/bin/env python3

import ihmm
import ihmm_io
import logging
import time
import numpy as np
import sys
import pyximport; pyximport.install()
from Sampler import *
from HmmSampler import *

class FullDepthCompiler():
    def __init__(self, depth):
        self.depth = depth
        
    def compile_models(self, models):
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        (a_max, b_max, g_max) = getVariableMaxes(models)
        totalK = get_state_size(models)
    
        t0 = time.time()
        pi = np.zeros((totalK, totalK))
        phi = np.zeros((totalK, models.lex.dist.shape[1]))
        
        ## Take exponent out of inner loop:
        word_dist = 10**models.lex.dist
    
        unlog_models(models, self.depth)

        for prevState in range(0,totalK):
            (prevF, prevJ, prevA, prevB, prevG) = extract_states(prevState, totalK, self.depth, getVariableMaxes(models))
            cumProbs = np.zeros(5)
            next_state = ihmm.State(self.depth)
            
            ## Some of the transitions look at above states, but in the d=0 special
            ## case there is nowhere to look so just assign the variable here and
            ## handle the special case outside of the loop:
            start_depth = get_cur_awa_depth(prevB)
            if start_depth == 0:
                above_act = 0
                above_awa = 0
            else:
                above_act = prevA[start_depth-1]
                above_awa = prevB[start_depth-1]
            
            for f in (0,1):
                nextState.f = [-1 for x in nextState.f]
                nextState.f[start_depth] = f
                cumProbs[0] = models.fork[start_depth].dist[ prevB[start_depth], prevG, f ]
                
                for j in (0,1):
                    nextState.j = [-1 for x in nextState.j]
                    nextState.j[start_depth] = j
                    if f == 0:
                        cumProbs[1] = cumProbs[0] * models.reduce[start_depth].dist[ prevA[start_depth], j ]
                    else:
                        cumProbs[1] = cumProbs[0] * models.trans[start_depth].dist[ prevB[start_depth], prevG, j ]
                    
                    for a in range(1, a_max-1):
                        for b in range(1, b_max-1):
                            nextState.a = [0 for x in nextState.a]
                            nextState.b = [0 for x in nextState.b]
                            
                            if f == 1 and j == 1:
                                if a == prevA[start_depth]:
                                    cumProbs[2] = cumProbs[1]
                                    for d in range(0, start_depth):
                                        nextState.a[d] = prevA[d]
                                        nextState.b[d] = prevB[d]
                                    nextState.a[start_depth] = a
                                    
                                    cumProbs[3] = cumProbs[2] * models.cont[start_depth].dist[ prevB[start_depth], prevG, b]
                                    nextState.b[start_depth] = b
                                    
                                else:
                                    continue
                            elif f == 0 and j == 1:
                                ## -/+ case, reducing a level
                                if start_depth == 0:
                                    continue
                                for d in range(0, start_depth-1):
                                    nextState.a[d] = prevA[d]
                                    nextState.b[d] = prevB[d]
                                
                                if a == prevA[start_depth-1]:
                                    cumProbs[2] = cumProbs[1]
                                    nextState.a[start_depth-1] = a
                                    cumProbs[3] = cumProbs[2] * models.next[start_depth-1].dist[ above_awa, prevA[state_depth], b ]
                                else:
                                    continue
                            elif f == 0 and j == 0:
                                ## -/-, reduce in place
                                for d in range(0, start_depth):
                                    nextState.a[d] = prevA[d]
                                    nextState.b[d] = prevB[d]
                                nextState.a[start_depth] = a
                                cumProbs[2] = cumProbs[1] * models.act[start_depth].dist[ prevA[start_depth], above_awa, a]
                                
                                nextState.b[start_depth] = b
                                cumProbs[3] = cumProbs[2] * models.start[start_depth].dist[ prevA[start_depth], a, b ]
                            elif f == 1 and j == 0:
                                ## +/-, create a new stack level
                                if start_depth+1 == self.depth:
                                    continue
                            
                                for d in range(0, start_depth+1):
                                    nextState.a[d] = prevA[d]
                                    nextState.b[d] = prevB[d]
                                nextState.a[start_depth+1] = a
                                cumProbs[2] = cumProbs[1] * models.root[start_depth+1].dist[ aboveAwa, prevG, a ]
                                
                                nextState.b[start_depth+1] = b
                                cumProbs[3] = cumProbs[2] * models.exp[start_depth+1].dist[ a, prevG, b ]
                                                        
                            ## Now multiply in the pos tag probability:
                            state_range = getStateRange(nextState.f, nextState.j, nextState.a, nextState.b, getVariableMaxes(models))         
                            range_probs = cumProbs[3] * (models.pos.dist[b,:-1])
                            pi[prevState, state_range] = range_probs            
                            
        time_spent = time.time() - t0
        logging.info("Done in %d s" % time_spent)
        relog_models(models, self.depth)
        return (pi, phi)

def unlog_models(models, depth):

    for d in range(0, depth):
        models.fork[d].dist = 10**models.fork[d].dist
        
        models.reduce[d].dist = 10**models.reduce[d].dist
        models.trans[d].dist = 10**models.trans[d].dist
        
        models.act[d].dist = 10**models.act[d].dist
        models.root[d].dist = 10**models.root[d].dist
        
        models.cont[d].dist = 10**models.cont[d].dist
        models.exp[d].dist = 10**models.exp[d].dist
        models.next[d].dist = 10**models.next[d].dist
        models.start[d].dist = 10**models.start[d].dist
        
    pos = 10**models.pos.dist

def relog_models(models, depth):
    for d in range(0, depth):
        models.fork[d].dist = np.log10(models.fork[d].dist)
        
        models.reduce[d].dist = np.log10(models.reduce[d].dist)
        models.trans[d].dist = np.log10(models.trans[d].dist)
        
        models.act[d].dist = np.log10(models.act[d].dist)
        models.root[d].dist = np.log10(models.root[d].dist)
        
        models.cont[d].dist = np.log10(models.cont[d].dist)
        models.exp[d].dist = np.log10(models.exp[d].dist)
        models.next[d].dist = np.log10(models.next[d].dist)
        models.start[d].dist = np.log10(models.start[d].dist)
        
    pos = np.log10(models.pos.dist)

def get_cur_awa_depth(stack):
    for d in range(len(stack)):
        if stack[d] > 0:
            return d

    return 0
