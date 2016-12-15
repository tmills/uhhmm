# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport cython
import logging
import pickle
import time
import numpy as np
cimport numpy as np
import zmq
import scipy.sparse
cimport Indexer
import Indexer
import Sampler
cimport Sampler
import State
cimport State
import subprocess
from uhhmm_io import printException
import models
cimport models

def boolean_depth(l):
    for index,val in enumerate(l):
        if val >= 0:
            return index

cdef class HmmSampler(Sampler.Sampler):
    
    def __init__(self, seed):
        Sampler.Sampler.__init__(self, seed)
        self.indexer = None
        self.models = None
#        self.pi = None
        
    def set_models(self, models):
        self.models = models[0]
        unlog_models(self.models)
        self.lexMatrix = np.matrix(self.models.lex.dist, copy=False)
        self.depth = len(self.models.fork)
        self.indexer = Indexer.Indexer(self.models)
        
        g_len = self.models.pos.dist.shape[1]
        w_len = self.models.lex.dist.shape[1]
        lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() / g_len)))
        self.data = lexMultiplier.data
        self.indices = lexMultiplier.indices
        self.indptr = lexMultiplier.indptr
        
    def initialize_dynprog(self, batch_size, maxLen):
        ## We ignore batch size since python only processes one at a time
        #self.dyn_prog = np.zeros((self.indexer.get_state_size(), maxLen))
        self.dyn_prog = np.zeros((maxLen, self.indexer.get_state_size()))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef forward_pass(self, pi, list sents, int sent_index):
        cdef float sentence_log_prob = 0, normalizer
        cdef double t0, t1
        cdef int a_max, b_max, g_max, index, token, g_len
        cdef tuple maxes
        cdef list sent
        
        if len(sents) > 1:
            raise Exception("Error: Python version only accepts batch size 1")
        
        sent = sents[0]
        t0 = time.time()
        try:           
            ## keep track of forward probs for this sentence:
            maxes = self.indexer.getVariableMaxes()
            (a_max,b_max,g_max) = maxes

            ## Copy=False indicates that this matrix object is just a _view_ of the
            ## array -- we don't have to copy it into a matrix and recopy back to array
            ## to get the return value
            self.dyn_prog[:] = 0
            lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
            ## Make forward be transposed up front so we don't have to do all the transposing inside the loop
            forward = np.matrix(self.dyn_prog, copy=False)
            
            for index,token in enumerate(sent):
                ## Still use special case for 0
                if index == 0:
                    forward[0,1:g_max-1] = self.lexMatrix[1:-1,token].transpose()

                else:
                    forward[index,:] = forward[index-1,:] * pi
                    
                    expanded_lex = self.lexMatrix[:,token].transpose() * lexMultiplier
                    forward[index,:] = np.multiply(forward[index,:], expanded_lex)

                normalizer = forward[index,:].sum()
                forward[index,:] /= normalizer
            
                ## Normalizer is p(y_t)
                sentence_log_prob += np.log10(normalizer)
            last_index = len(sent)-1
            if np.argwhere(forward.max(1)[0:last_index+1,:] == 0).size > 0 or np.argwhere(np.isnan(forward.max(1)[0:last_index+1,:])).size > 0:
                logging.error("Error; There is a word with no positive probabilities for its generation in the forward filter: %s" % forward.max(1)[0:last_index+1,:])
                raise Exception("There is a word with no positive probabilities for its generation in the forward filter.")

            ## FIXME - Do we need to multiply by -/+ at last time step for d > 0 system?
            ## TODO - should we make probability of transitioning to d > 0 state at 
            ## last time step = 0.0? Or just sample it with prob = 0.0?
        except Exception as e:
            printException()
            raise e
            
        t1 = time.time()
        # print('forward time', t1-t0)
        self.ff_time += (t1-t0)
        return [sentence_log_prob]
   
    cpdef reverse_sample(self, pi, list sents, int sent_index):
        cdef int totalK, depth, last_index, sample_t, sample_depth, t, ind
        cdef int prev_depth, next_f_depth, next_awa_depth
        cdef float trans_prob, sample_log_prob
        cdef double t0, t1
        cdef list sample_seq, sent
        cdef tuple maxes
        cdef np.ndarray trans_slice
        cdef State.State sample_state
        
        sent = sents[0]
        t0 = time.time()
        try:      
            sample_seq = []
            sample_log_prob = 0
            maxes = self.indexer.getVariableMaxes()
            totalK = self.indexer.get_state_size()
            depth = len(self.models.fork)
        
            ## Normalize and grab the sample from the forward probs at the end of the sentence
            last_index = len(sent)-1
                
            ## normalize after multiplying in the transition out probabilities
            self.dyn_prog[last_index,:] /= self.dyn_prog[last_index,:].sum()
            
            sample_t = -1
            sample_depth = -1
            ## We require that the sample comes from the set of states that are at depth
            ## 0 (i.e. the sentence must fully reduce to be a valid parse)
            #print(dyn_prog[:,last_index])
            while sample_t < 0 or sample_depth > 0:
                sample_t = get_sample(self.dyn_prog[last_index,:])
                sample_state = self.indexer.extractState(sample_t)
                sample_depth = sample_state.max_awa_depth()
                #logging.debug("Sampled final state %s with depth %d" % (sample_state.str(), sample_depth))
    
            sample_seq.append(sample_state)
            #logging.debug("Sampled state %s at time %d" % (sample_seq[-1].str(), last_index))
            
            if last_index > 0 and (sample_seq[-1].a[0] == 0 or sample_seq[-1].b[0] == 0 or sample_seq[-1].g == 0):
                logging.error("Error: First sample for sentence %d has index %d and has a|b|g = 0" % (sent_index, sample_t))
                raise Exception
  
            for t in range(len(sent)-2,-1,-1):                
                sample_state, sample_t = self._reverse_sample_inner(pi, sample_t, t)
                sample_seq.append(sample_state)
           
            sample_seq.reverse()
        except Exception as e:
            printException()
            raise e
        
        t1 = time.time()
        # print('backward time', t1-t0)
        self.bs_time += (t1-t0)
        return [sample_seq]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _reverse_sample_inner(self, pi, int sample_t, int t):
        cdef np.ndarray trans_slice
        cdef int ind
        cdef float normalizer
        
        trans_slice = pi[:,sample_t].toarray()
        for ind in np.where(self.dyn_prog[t,:] != 0.0)[0]:                     
            self.dyn_prog[t,ind] *= trans_slice[ind]
        
        normalizer = self.dyn_prog[t,:].sum()
        if normalizer == 0.0:
            logging.warning("No positive probability states at this time step %d." % (t))
        
        self.dyn_prog[t,:] /= normalizer
        
        sample_t = get_sample(self.dyn_prog[t,:])
        sample_state = self.indexer.extractState(sample_t)
        #logging.log(logging.DEBUG-1, "Sampled state %s with index %d at time %d" % (sample_state.str(), sample_t, t))
        
        if t > 0 and sample_state.g == 0:
            logging.error("Error: Sampled a g=0 state with state index %d in backwards pass: %s" % (sample_t, sample_state.str()) )
            raise Exception
        
        return sample_state, sample_t
        
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int max_awa_depth(np.ndarray b):
    cdef int d = 0
    for d in range(0, len(b)):
        if b[d] == 0:
            return d-1
            
    return 0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # faster division without checks
cdef int get_sample(np.ndarray[np.float64_t] dist):
    cdef float dart
    cdef np.ndarray[np.float64_t] sum_dist
    cdef int start, end, cur, ret
    
    sum_dist = np.cumsum(dist)
    dart = np.random.random()
    start = 0
    end = len(dist) - 1
    cur = (end + start) / 2
    while True:
        if dart > sum_dist[cur]:
            if cur == end - 1:
                ret = cur+1
                break
                
            start = cur
        else:
            ## dart < sum_dist but need to check 3 cases:
            if cur == 0:
                ret = cur
                break
            elif sum_dist[cur-1] < dart:
                ret = cur
                break
            else:
                end = cur
        
        cur = (end+start) / 2

    return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int old_get_sample(np.ndarray dist):
    cdef np.ndarray sum_dist
    cdef int i
    cdef float dart
    
    sum_dist = np.cumsum(dist)
    dart = np.random.random()
    
    for i in range(0, len(dist)):
        if dart < sum_dist[i]:
            return i
    
def unlog_models(models):
    depth = len(models.fork)
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
    models.lex.dist = 10**models.lex.dist
