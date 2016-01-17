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

def compile_and_store_models(models, working_dir):
    scipy = True
    try:
        import scipy.sparse
    except:
        logging.warn("Could not find scipy! Using numpy will be much less memory efficient!")
        scipy = False

    logging.info("Compiling component models into mega-HMM transition and observation matrices")
    (a_max, b_max, g_max) = getVariableMaxes(models)
    totalK = get_state_size(models)
    
    t0 = time.time()
    pi = np.zeros((totalK, totalK))
        
    ## Take exponent out of inner loop:
    word_dist = 10**models.lex.dist
    
    (fork,act,root,cont,start,pos) = unlog_models(models)

    ## For previous state i:
    for prevState in range(0,totalK):
        (prevF, prevJ, prevA, prevB, prevG) = extractStates(prevState, totalK, getVariableMaxes(models))
        cumProbs = np.zeros(5)
        
        ## Sample f & j:
        for f in (0,1):
            if prevA == 0 and prevB == 0 and f == 0:
                continue
    
            cumProbs[0] = (fork[prevB, prevG,f])
    
            for j in (0,1):
                ## At depth 1 -- no probability model for j
                if prevA == 0 and prevB == 0:
                    ## index 1:
                    if j == 0:
                        cumProbs[1] = cumProbs[0]
                    else:
                        cumProbs[1] = 0
                        ## No point in continuing -- matrix is zero'd to start
                        continue
                
                elif f == j:
                    cumProbs[1] = cumProbs[0]
                else:
                    cumProbs[1] = 0
                    continue    
        
                for a in range(1,a_max-1):
                    if f == 0 and j == 0:
                        ## active transition:
                        cumProbs[2] = cumProbs[1] * (act[prevA,a])
                    elif f == 1 and j == 0:
                        ## root -- technically depends on prevA and prevG
                        ## but in depth 1 this case only comes up at start
                        ## of sentence and prevA will always be 0
                        cumProbs[2] = cumProbs[1] * (root[prevG,a])
                    elif f == 1 and j == 1 and prevA == a:
                        cumProbs[2] = cumProbs[1]
                    else:
                        ## zero probability here
                        continue
        
                    if cumProbs[2] == 0:
                        continue

                    for b in range(1,b_max-1):
                        if j == 1:
                            cumProbs[3] = cumProbs[2] * (cont[prevB, prevG, b])
                        else:
                            cumProbs[3] = cumProbs[2] * (start[prevA, a, b])
            
                        # Multiply all the g's in one pass:
                        ## range gets the range of indices in the forward pass
                        ## that are contiguous in the state space
                        state_range = getStateRange(f,j,a,b, getVariableMaxes(models))
                                     
                        range_probs = cumProbs[3] * (pos[b,:-1])
                        pi[prevState, state_range] = range_probs
                           
    time_spent = time.time() - t0
    logging.info("Done in %d s" % time_spent)
    if scipy:
        trans_mat = (scipy.sparse.csc_matrix(pi,copy=False)) #, np.matrix(phi,copy=False))
    else:
        trans_mat = (np.matrix(pi, copy=False), np.matrix(phi, copy=False))

    fn = working_dir+'/models.bin'
    out_file = open(fn, 'wb')
    model = PyzmqModel((models,trans_mat), finite=True)
    pickle.dump(model, out_file)
    out_file.close()
    
def unlog_models(models):
    fork = 10**models.fork[0].dist
    act = 10**models.act[0].dist
    root = 10**models.root[0].dist
    cont = 10**models.cont[0].dist
    start = 10**models.start[0].dist
    pos = 10**models.pos.dist
   
    return (fork,act,root,cont,start,pos)

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


class FiniteSampler(Sampler):
                    
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
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        ## normalize after multiplying in the transition out probabilities
        dyn_prog[:,last_index] /= dyn_prog[:,last_index].sum()
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                        
        sample_seq.append(ihmm.State(1, extractStates(sample_t, totalK, maxes)))
        if last_index > 0 and (sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0):
            logging.error("Error: First sample has a|b|g = 0")
            raise ParseException("Error: First sample has a|b|g = 0")
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == 0:
                    continue

                (pf,pj,pa,pb,pg) = extractStates(ind, totalK, getVariableMaxes(self.models))
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
            state_list = extractStates(sample_t, totalK, getVariableMaxes(self.models))
            
            sample_state = ihmm.State(1, state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
            sample_seq.append(sample_state)
            
        sample_seq.reverse()
        logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq

