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
import scipy.sparse

class FullDepthCompiler():
    def __init__(self, depth):
        self.depth = depth
        
    def compile_models(self, models):
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        maxes = getVariableMaxes(models)
        (a_max, b_max, g_max) = maxes
        totalK = get_state_size(models)
        
        cache_hits = 0
        t0 = time.time()
        #pi = np.zeros((totalK, totalK))
        #pi = scipy.sparse.csr_matrix((totalK, totalK))
        indptr = np.zeros(totalK+1)
        indices =  []
        data = []
        
        ## Take exponent out of inner loop:
        word_dist = 10**models.lex.dist
        unlog_models(models, self.depth)
                
        for prevIndex in range(0,totalK):
#             if prevIndex % 1000 == 0:
#                 logging.info("prevIndex index=%d after %d s for %f"  % (prevIndex, time.time()-t0, prevIndex/totalK))
            (prevF, prevJ, prevA, prevB, prevG) = extractStacks(prevIndex, totalK, self.depth, maxes)
            
            ## Skip invalid start states
            if self.depth > 1:
                ## One that should not be allowed by our indexing scheme:
                for d in range(0, self.depth):
                    if len(np.where(prevF >= 0)[0]) > 1 or len(np.where(prevJ >= 0)[0]) > 1:                      
                        logging.error("Two values in F stack with nonnegative value: this should not be allowed")
                        raise Exception

                ## Others that are allowed but not really possible:
                ## 1) where lower depths have real values but higher depths have
                ## placeholder 0 values
                ## 2) where one of A and B has a placeholder at the same depth level
                lowest_depth = -1
                for d in range(self.depth-1, -1, -1):
                    if(prevA[d] * prevB[d] == 0 and prevA[d] + prevB[d] != 0):
                        continue
                    if lowest_depth == -1 and prevA[d] > 0:
                        ## Until we find a non-zero value move the depth up
                        lowest_depth = d
                    else:
                        ## we have seen a non-zero value deeper than us, so we can't see any zeros now
                        if prevA[d] == 0 or prevB[d] == 0:
                            continue
            
            cumProbs = np.zeros(5)
            nextState = ihmm.State(self.depth)
            
            ## Some of the transitions look at above states, but in the d=0 special
            ## case there is nowhere to look so just assign the variable here and
            ## handle the special case outside of the loop:
            start_depth = get_cur_awa_depth(prevB)
            if start_depth == -1:
                above_act = 0
                above_awa = 0
            else:
                above_act = prevA[start_depth-1]
                above_awa = prevB[start_depth-1]
            
#             cache_key = (start_depth, above_awa, prevA[start_depth], prevB[start_depth], prevG)
#             
#             if cache_key in cache:
#            if cache_index[start_depth][above_awa][ prevA[start_depth] ][ prevB[start_depth] ][prevG] == 1:
#                cache_hits += 1
#                if cache_hits % 1000 == 0:
#                    logging.info("%d cache hits" % cache_hits)
#                t00 = time.time()
#                pi[prevIndex] = cache[start_depth][above_awa][ prevA[start_depth] ][ prevB[start_depth] ][prevG]
#                pi[prevIndex] = cache[cache_key]
#                pi[prevIndex] = pi[ cache[start_depth][above_awa][ prevA[start_depth] ][ prevB[start_depth] ][prevG] ]
#                t01 = time.time()
#                #logging.info("Cache fill at prevIndex took %f s" % (t01-t00))
#                continue

            t00 = time.time()
            for f in (0,1):
                nextState.f[:] = -1
                nextState.f[start_depth] = f
                cumProbs[0] = models.fork[start_depth].dist[ prevB[start_depth], prevG, f ]
                for j in (0,1):
                    if prevG == 0 and not (f == 1 and j == 0):
                        continue
                        
                    nextState.j[:] = -1
                    nextState.j[start_depth] = j
                    if f == 0:
                        cumProbs[1] = cumProbs[0] * models.reduce[start_depth].dist[ prevA[start_depth], above_awa, j ]
                    else:
                        cumProbs[1] = cumProbs[0] * models.trans[start_depth].dist[ above_awa, prevG, j ]
                    
                    for a in range(1, a_max-1):
                        nextState.a[:] = 0
                        for b in range(1, b_max-1):
                            nextState.b[:] = 0
                            
                            if f == 1 and j == 1:
                                if a == prevA[start_depth]:
                                    cumProbs[2] = cumProbs[1]
                                    nextState.a[0:start_depth] = prevA[0:start_depth]
                                    nextState.b[0:start_depth] = prevB[0:start_depth]
                                    
                                    nextState.a[start_depth] = a
                                    
                                    cumProbs[3] = cumProbs[2] * models.cont[start_depth].dist[ prevB[start_depth], prevG, b]
                                    nextState.b[start_depth] = b
                                    
                                else:
                                    continue
                            elif f == 0 and j == 1:
                                ## -/+ case, reducing a level
                                if start_depth <= 0:
                                    continue
                                nextState.a[0:start_depth-1] = prevA[0:start_depth-1]
                                nextState.b[0:start_depth-1] = prevB[0:start_depth-1]
                                
                                if a == prevA[start_depth-1]:
                                    cumProbs[2] = cumProbs[1]
                                    nextState.a[start_depth-1] = a
                                    cumProbs[3] = cumProbs[2] * models.next[start_depth-1].dist[ above_awa, nextState.a[start_depth-1], b ]
                                else:
                                    continue
                            elif f == 0 and j == 0:
                                ## -/-, reduce in place
                                nextState.a[0:start_depth] = prevA[0:start_depth]
                                nextState.b[0:start_depth] = prevB[0:start_depth]
                                nextState.a[start_depth] = a
                                cumProbs[2] = cumProbs[1] * models.act[start_depth].dist[ prevA[start_depth], above_awa, a]
                                
                                nextState.b[start_depth] = b
                                cumProbs[3] = cumProbs[2] * models.start[start_depth].dist[ prevA[start_depth], a, b ]
                            elif f == 1 and j == 0:
                                ## +/-, create a new stack level
                                if start_depth+1 == self.depth:
                                    continue
                            
                                nextState.a[0:start_depth+1] = prevA[0:start_depth+1]
                                nextState.b[0:start_depth+1] = prevB[0:start_depth+1]
                                nextState.a[start_depth+1] = a
                                cumProbs[2] = cumProbs[1] * models.root[start_depth+1].dist[ prevB[start_depth], prevG, a ]
                                
                                nextState.b[start_depth+1] = b
                                cumProbs[3] = cumProbs[2] * models.exp[start_depth+1].dist[ prevG, nextState.a[start_depth+1], b ]
                                                        
                            ## Now multiply in the pos tag probability:
                            state_range = getStateRange(nextState.f, nextState.j, nextState.a, nextState.b, maxes)         
                            range_probs = cumProbs[3] * (models.pos.dist[b,:-1])
                            indptr[prevIndex+1] += len(range_probs)
                            
                            for g in range(0,len(range_probs)):
                                indices.append(state_range[g])
                                data.append(range_probs[g])
                                
                            #pi[prevIndex, state_range] = range_probs

            ## now that we've finished for this prevIndex cache it:
            #cache[cache_key] = pi[prevIndex,:]
#             cache[start_depth][above_awa][ prevA[start_depth] ][ prevB[start_depth] ][prevG] = prevIndex
#             cache_index[start_depth][above_awa][ prevA[start_depth] ][ prevB[start_depth] ][prevG] = 1
            t01 = time.time()
#            if np.random.random() < 0.01:
#                logging.info("non-cache filling took %f s" % (t01-t00))

        time_spent = time.time() - t0
        logging.info("Done in %d s with %d cache hits" % (time_spent, cache_hits))
        relog_models(models, self.depth)
        return scipy.sparse.csr_matrix((data,indices,indptr), (totalK, totalK), dtype=np.float64)

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
        
    models.pos.dist = 10**models.pos.dist

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
        
    models.pos.dist = np.log10(models.pos.dist)

def get_cur_awa_depth(stack):
    for d in range(len(stack)):
        if stack[d] > 0:
            return d

    return -1