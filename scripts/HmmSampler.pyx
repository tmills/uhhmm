# cython: profile=False
# cython: linetrace=True
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
#cimport Sampler
import State
cimport State
import subprocess
from uhhmm_io import printException
import models
cimport models
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import skcuda
linalg.init()

def boolean_depth(l):
    for index,val in enumerate(l):
        if val >= 0:
            return index

class HmmSampler(Sampler.Sampler):
    
    def __init__(self, seed):
        Sampler.Sampler.__init__(self, seed)
        self.indexer = None
        self.models = None
#        self.pi = None
        
    def set_models(self, models):
        self.models = models[0]
        unlog_models(self.models)
        #self.lexMatrix = gpuarray.to_gpu( np.expand_dims(self.models.lex.dist.astype('float32'), 0) )
        self.lexMatrix = self.models.lex.dist.astype('float32')
        
        self.depth = len(self.models.fork)
        self.indexer = Indexer.Indexer(self.models)
        
        g_len = self.models.pos.dist.shape[1]
        w_len = self.models.lex.dist.shape[1]
        self.lexMultiplier = gpuarray.to_gpu(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() / g_len)).astype('float32'))
        
    def initialize_dynprog(self, maxLen):
        self.maxLen = maxLen

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward_pass(self, pi_gpu, list sents, int sent_index):
        cdef float sentence_log_prob = 0, normalizer
        cdef double t0, t1
        cdef int a_max, b_max, g_max, index, token, g_len
        cdef tuple maxes
                
        batch_size = len(sents)
        batch_max_len = max(map(len, sents))
        np_sents = get_sentence_array(sents, batch_max_len)
        
        logging.info("Processing a batch of size %d with max length %d" % (batch_size, batch_max_len) )
        sentence_log_probs = np.zeros( len(sents) )
        
        t0 = time.time()
        try:           
            ## keep track of forward probs for this sentence:
            maxes = self.indexer.getVariableMaxes()
            (a_max,b_max,g_max) = maxes

            forward = gpuarray.zeros( (batch_max_len, batch_size, self.indexer.state_size), np.float32)
            #forward = gpuarray.zeros( (batch_size, self.indexer.state_size, batch_max_len, np.float32)
            ones = gpuarray.zeros( (batch_size, g_max-2), np.float32) + 1
            ones_mat = gpuarray.zeros( (batch_size, self.indexer.state_size), np.float32 ) + 1
            normalizers = gpuarray.zeros( (batch_size, self.indexer.state_size), np.float32 )
            
            #gpu_lex = gpuarray.to_gpu(self.lexMatrix)
            
            #lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
            #next = gpuarray.zeros((1, self.indexer.state_size), dtype='float32')
            
            for index in range(batch_max_len):
                logging.info("Index=%d" % (index) )
                ## First do transition in a batch:
                ## Still use special case for 0
                if index == 0:
                    forward[0,:,1:g_max-1] = ones  #self.lexMatrix[:,1:-1,token]
                else:
                    forward[index] = linalg.dot(forward[index-1], pi_gpu)
                
                #logging.info("max is %s and sum is %s" % ( skcuda.misc.max(forward[index], 1), skcuda.misc.sum(forward[index], 1) ) )
                
                ## TODO -- make the inputs to this already gpu arrays so we don't have to convert every time
                #expanded_lex = gpuarray.to_gpu( (self.lexMatrix[:,token].transpose() * lexMultiplier).astype(np.float32) )
                #logging.info("shape of lex col=%s, lex multiplier =%s" % ( str(self.lexMatrix[:,:,token].shape), str(self.lexMultiplier.shape) ) )
                
                ## Get the column vector representing the token index of every sentence in the corpus at current index:
                tokens = np_sents[:,index]
                
                gpu_lex = gpuarray.to_gpu( self.lexMatrix[:,tokens].transpose() )
                
                #logging.info("Shape of tokens=%s, lexMultipler=%s, gpu_lex=%s" % ( str(tokens.shape), str(self.lexMultiplier.shape), str(gpu_lex.shape) ) )
                
                expanded_lex = linalg.dot(gpu_lex, self.lexMultiplier)
                    
                #logging.info("Shape of forward section=%s, expanded_lex=%s" % ( str(forward[index].shape), str(expanded_lex.shape) ) )
                forward[index] = linalg.multiply(forward[index], expanded_lex)                       

                #logging.info("max is %s and sum is %s" % ( skcuda.misc.max(forward[index], 1), skcuda.misc.sum(forward[index], 1) ) )

                sums = skcuda.misc.sum( forward[index], 1 )
                
                ## One normalizer per sentence at each token:
                assert len(sums) == batch_size
                
                ## FIXME -- mult_matvec is buggy
                #normalizers = skcuda.misc.mult_matvec(ones_mat.transpose(), 1.0 / sums).transpose()
                for sent_ind in range(batch_size):
                    forward[index,sent_ind] /= sums[sent_ind].get()
                                
                #logging.info("Sums=%s" % (sums) )
                
                #logging.info("Sums after norm (should be 1s)=%s" % (skcuda.misc.sum( forward[index], 1 )) )
                           
                ## Normalizer is p(y_t)
                sentence_log_probs += np.log10(sums.get())

            ## FIXME - for some reason this doesn't work with pycuda max.
            #if np.argwhere(skcuda.misc.max(forward,1)[0:last_index+1] == 0).size > 0:
                #logging.error("Error; There is a word with no positive probabilities for its generation in the forward filter: %s" % skcuda.misc.max(forward, 1)[0:last_index+1])
                #raise Exception("There is a word with no positive probabilities for its generation in the forward filter.")

            ## FIXME - for some reason this doesn't work with pycuda max.
            #if np.argwhere(np.isnan(skcuda.misc.max(forward, 1)[0:last_index+1])).size > 0:
            #    logging.error("Error; There is a word with no positive probabilities for its generation in the forward filter: %s" % skcuda.misc.max(forward, 1)[0:last_index+1])
            #    raise Exception("There is a word with nan probabilities for its generation in the forward filter.")

            ## FIXME - Do we need to multiply by -/+ at last time step for d > 0 system?
            ## TODO - should we make probability of transitioning to d > 0 state at 
            ## last time step = 0.0? Or just sample it with prob = 0.0?
        except Exception as e:
            printException()
            raise e
        
        #debug_array = forward[ len(sents[0])-1, 0, :].get()
        #logging.info("End of forward, sum of values is %f and max is %f" % ( debug_array.sum(), debug_array.max() ) )

        t1 = time.time()
        self.ff_time += (t1-t0)
        return sentence_log_probs, forward
   
    def reverse_sample(self, forward, gpu_pi, list sents, int sent_index):
        cdef int totalK, depth, last_index, sample_t, sample_depth, t, ind, num_sents
        cdef int prev_depth, next_f_depth, next_awa_depth
        cdef float trans_prob, sample_log_prob
        cdef double t0, t1
        cdef list sample_seq
        cdef tuple maxes
        cdef np.ndarray trans_slice
        cdef State.State sample_state
        cdef list sample_seqs
        
        num_sents = forward.shape[1]
        sample_seqs = []
        
        ## Convert back to numpy, get a copy with fortran ordering, then copy back to GPU:
        gpu_pi_f = gpuarray.to_gpu( gpu_pi.get().copy('f') )

        #debug_array = forward[ len(sents[0])-1, 0, :].get()
        #logging.info("For first sentence, sum of values is %f and max is %f" % ( debug_array.sum(), debug_array.max() ) )
        
        t0 = time.time()
        try:
#            for ind in range(sent_index, sent_index+num_sents):
            for ind in range( num_sents ):
                sample_seq = []
                sample_log_prob = 0
                maxes = self.indexer.getVariableMaxes()
                totalK = self.indexer.get_state_size()
                depth = len(self.models.fork)
        
                ## Normalize and grab the sample from the forward probs at the end of the sentence
                last_index = len(sents[ind])-1
            
                #logging.info("Attempting to sample sentence %d with length %d and last index %d from forward matrix with shape %s" % (ind, len(sents[ind]), last_index, str(forward.shape) ) )
                ## normalize after multiplying in the transition out probabilities
                #forward[last_index,:] /= skcuda.misc.sum(forward[last_index,:]) #.sum()
            
                sample_t = -1
                sample_depth = -1
                ## We require that the sample comes from the set of states that are at depth
                ## 0 (i.e. the sentence must fully reduce to be a valid parse)
                #print(dyn_prog[:,last_index])
                while sample_t < 0 or sample_depth > 0:
                    #sample_t = get_gpu_sample(forward[last_index, ind, :])
                    sample_t = get_sample(forward[last_index, ind, :].get() )
                    sample_state = self.indexer.extractState(sample_t)
                    sample_depth = sample_state.max_awa_depth()
                    #logging.info("Sampled final state %s with depth %d" % (sample_state.str(), sample_depth))
    
                sample_seq.append(sample_state)
            
                #if last_index > 0 and (sample_seq[-1].a[0] == 0 or sample_seq[-1].b[0] == 0 or sample_seq[-1].g == 0):
                #    logging.error("Error: First sample for sentence %d has index %d and has a|b|g = 0" % (sent_index, sample_t))
                #    raise Exception
  
                for t in range(len(sents[ind])-2,-1,-1):                
                    sample_state, sample_t = self._reverse_sample_inner(forward[:,ind,:], gpu_pi_f, sample_t, t)
                    sample_seq.append(sample_state)
                    #logging.info("Sampled state %s at time %d" % (sample_seq[-1].str(), t))
           
                sample_seq.reverse()
                sample_seqs.append(sample_seq)
        except Exception as e:
            printException()
            raise e
        
        t1 = time.time()
        self.bs_time += (t1-t0)
        return sample_seqs

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _reverse_sample_inner(self, forward, gpu_pi, int sample_t, int t):
        cdef np.ndarray trans_slice
        cdef int ind
        cdef float normalizer
        
        try:
        #logging.info("Before multiplying in transitions, forward = %s"  % ( str(forward[t,:]) ) )
        
        #for ind in range(len(forward[t,:])):
        #    forward[t,ind] *= gpu_pi[ind, sample_t]
            
        #logging.info("After multiplying in transitions, forward = %s"  % ( str(forward[t,:]) ) )

        #trans_probs = gpuarray.zeros( gpu_pi.shape, np.float32) + gpu_pi[:, sample_t]
            forward[t,:] = linalg.multiply(forward[t,:], gpu_pi[:,sample_t])
        
        ## FIXME -- this product seems to be buggy!
        #forward[t,0,:] = linalg.multiply(forward[t,:,:], gpu_pi[:, :, sample_t])
        
            normalizer = float(skcuda.misc.sum(forward[t,:]).get())
        #logging.info("Value of noramlizer is %s" % str(normalizer) )
        
        #if normalizer == 0.0:
        #    logging.warning("No positive probability states at this time step %d." % (t))
        
            forward[t,:] /= normalizer
        
        #sample_t = get_gpu_sample(forward[t,:])
            sample_t = get_sample(forward[t,:].get() )
            sample_state = self.indexer.extractState(sample_t)
        #logging.info("Sampled state %s with index %d at time %d" % (sample_state.str(), sample_t, t))
    
        except Exception as e:
            printException()
            raise e
                
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
cdef int get_sample(np.ndarray[np.float32_t] dist):
    cdef float dart
    cdef np.ndarray[np.float32_t] sum_dist
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
@cython.cdivision(True)    # faster division without checks
## TODO - find out what pycuda types are if I want to type this
cdef get_gpu_sample(dist):
    cdef float dart
    cdef int i
    sum_dist = skcuda.misc.cumsum(dist)
    dart = np.random.random()
   
    for i in range(0, len(dist)):
        if dart < sum_dist[i].get():
            return i
    
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

def get_sentence_array(list sents, int max_len):
    np_sents = np.zeros( (len(sents), max_len), dtype=np.int )
    
    for ind,sent in enumerate(sents):
        copy_sent = sent.copy()
        while len(copy_sent) < max_len:
            copy_sent.append(0)
        
        np_sents[ind] += copy_sent
    
    return np_sents
    
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
