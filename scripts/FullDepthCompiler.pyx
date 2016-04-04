# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1

cimport cython
import cython
import logging
import time
import numpy as np
cimport numpy as np
import sys
#import pyximport; pyximport.install()
from State import State
from Sampler import *
from HmmSampler import *
from PyzmqMessage import *
#from Indexer import Indexer
import Indexer
import scipy.sparse

cdef class FullDepthCompiler:
    cdef int depth
    
    def __init__(self, depth):
        self.depth = depth
    
    #@profile
    cdef compile_one_line(self, int prevIndex, models, indexer):
        cdef int totalK, a_max, b_max, g_max, start_depth, above_act, above_awa, state_index
        cdef int f,j,a,b,g
        cdef np.ndarray range_probs, prevF, prefJ, prevA, prevB
        #cdef np.ndarray indices, data
        
        prev_state = indexer.extractState(prevIndex)
        (prevF, prevJ, prevA, prevB, prevG) = prev_state.f, prev_state.j, prev_state.a, prev_state.b, prev_state.g
        
        #(prevF, prevJ, prevA, prevB, prevG) = indexer.extractStacks(prevIndex)
        start_depth = get_cur_awa_depth(prevB)
        (a_max, b_max, g_max) = indexer.getVariableMaxes()
        totalK = indexer.get_state_size()        
        indices =  []
        data = []

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
                    ## Exactly one of the values at this depth is zero -- not allowed
                    return indices, data
                if lowest_depth == -1:
                    if prevA[d] > 0:
                        ## Until we find a non-zero value move the depth up
                        lowest_depth = d
                else:
                    ## we have seen a non-zero value deeper than us, so we can't see any zeros now
                    if prevA[d] == 0 or prevB[d] == 0:
                        return indices, data
    
        cumProbs = np.zeros(3)
        nextState = State(self.depth)
    
        ## Some of the transitions look at above states, but in the d=0 special
        ## case there is nowhere to look so just assign the variable here and
        ## handle the special case outside of the loop:
        if start_depth <= 0:
            above_act = 0
            above_awa = 0
        else:
            above_act = prevA[start_depth-1]
            above_awa = prevB[start_depth-1]
    
        t00 = time.time()
        for f in (0,1):
            nextState.f[:] = -1
            ## when t=0, start_depth will be -1, which in the d> 1 case will wraparound.
            ## we want in the t=0 case for f_{t=1} to be [-/-]*d
            if start_depth >= 0:
                nextState.f[start_depth] = f
                cumProbs[0] = models.fork[start_depth].dist[ prevB[start_depth], prevG, f ]
            else:
                ## We only need to handle the f=-1 case one time, when f=1 and j=0 (though formally they are -1 and -1 because the 1 and 0 are above the stack we actually keep track of)
                if f == 1:
                    cumProbs[0] = 1.0
                else:
                    continue
                
            for j in (0,1):
                if start_depth == -1 and not (f == 1 and j == 0):
                    continue

                nextState.j[:] = -1
                ## See note above where we set nextState.f
                if start_depth >= 0:
                    nextState.j[start_depth] = j
                    if f == 0:
                        cumProbs[1] = cumProbs[0] * models.reduce[start_depth].dist[ prevA[start_depth], above_awa, j ]
                    else:
                        cumProbs[1] = cumProbs[0] * models.trans[start_depth].dist[ prevB[start_depth], prevG, j ]
                else:
                    if j == 0:
                        cumProbs[1] = 1
                    else:
                        ## Should be dead code from above
                        continue
                
                for a in range(1, a_max-1):
                    nextState.a[:] = 0
                    for b in range(1, b_max-1):
                        nextState.b[:] = 0

                        if f == 1 and j == 1:
                            if a == prevA[start_depth]:
                                nextState.a[0:start_depth] = prevA[0:start_depth]
                                nextState.b[0:start_depth] = prevB[0:start_depth]
                            
                                nextState.a[start_depth] = a
                            
                                cumProbs[2] = cumProbs[1] * models.cont[start_depth].dist[ prevB[start_depth], prevG, b]
                                nextState.b[start_depth] = b
                            
                            else:
                                continue
                        elif f == 0 and j == 1:
                            ## -/+ case, reducing a level unless already at minimum depth
                            if start_depth <= 0:
                                continue
                            nextState.a[0:start_depth-1] = prevA[0:start_depth-1]
                            nextState.b[0:start_depth-1] = prevB[0:start_depth-1]
                        
                            if a == prevA[start_depth-1]:
#                                cumProbs[2] = cumProbs[1]
                                nextState.a[start_depth-1] = a
                                nextState.b[start_depth-1] = b
                                cumProbs[2] = cumProbs[1] * models.next[start_depth-1].dist[ prevA[start_depth], above_awa, b ]
                            else:
                                continue
                        elif f == 0 and j == 0:
                            ## -/-, reduce in place
                            nextState.a[0:start_depth] = prevA[0:start_depth]
                            nextState.b[0:start_depth] = prevB[0:start_depth]
                            nextState.a[start_depth] = a
                            cumProbs[2] = cumProbs[1] * models.act[start_depth].dist[ prevA[start_depth], above_awa, a] * models.start[start_depth].dist[ prevA[start_depth], a, b ]
                        
                            nextState.b[start_depth] = b
                        elif f == 1 and j == 0:
                            ## +/-, create a new stack level unless we're at the limit
                            if start_depth+1 == self.depth:
                                continue
                    
                            nextState.a[0:start_depth+1] = prevA[0:start_depth+1]
                            nextState.b[0:start_depth+1] = prevB[0:start_depth+1]
                            nextState.a[start_depth+1] = a
                            cumProbs[2] = cumProbs[1] * models.root[start_depth+1].dist[ above_awa, prevG, a ] * models.exp[start_depth+1].dist[ prevG, a, b ]
                        
                            nextState.b[start_depth+1] = b
                                                
                        ## Now multiply in the pos tag probability:
                        state_index = indexer.getStateIndex(nextState.f, nextState.j, nextState.a, nextState.b, 0)         
                        range_probs = cumProbs[2] * (models.pos.dist[b,:-1])
                        #logging.info("Building model with %s => %s" % (prev_state.str(), nextState.str() ) )
                        for g in range(1,len(range_probs)):
                            indices.append(state_index + g)
                            data.append(range_probs[g])

        return indices, data
    
    #@profile
    def compile_and_store_models(self, models, working_dir):
        indexer = Indexer.Indexer(models)
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        maxes = indexer.getVariableMaxes()
        (a_max, b_max, g_max) = maxes
        totalK = indexer.get_state_size()
        
        cache_hits = 0
        t0 = time.time()
        indptr = np.zeros(totalK+1)
        indices =  []
        data = []

        ## Take exponent out of inner loop:
        unlog_models(models, self.depth)
                
        for prevIndex in range(0,totalK):
            indptr[prevIndex+1] = indptr[prevIndex]
            (local_indices, local_data) = self.compile_one_line(prevIndex, models, indexer)
            indptr[prevIndex+1] += len(local_indices)
            indices.append(local_indices)
            data.append(local_data)

        logging.info("Flattening sublists into main list")
        flat_indices = [item for sublist in indices for item in sublist]
        flat_data = [item for sublist in data for item in sublist]
            
        relog_models(models, self.depth)
        logging.info("Creating csr transition matrix from sparse indices")
        pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK), dtype=np.float64)
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        logging.info("Transforming and writing csc model")
        model = PyzmqModel((models,pi.tocsc()), finite=True)
        pickle.dump(model, out_file)
        out_file.close()
        nnz = pi.nnz
        pi = None

        time_spent = time.time() - t0
        logging.info("Done in %d s with %d cache hits and %d non-zero values" % (time_spent, cache_hits, nnz))
        

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
    ## Essentially empty -- used for first time step
    if stack[0] <= 0:
        return -1
    
    ## If we encounter a zero at position 1, then the depth is 0
    for d in range(1, len(stack)):
        if stack[d] == 0:
            return d-1
    
    ## Stack is full -- if d=4 then max depth index is 3
    return len(stack)-1
