#!/usr/bin/env python3

import ihmm
import ihmm_io
import logging
import time
import numpy as np
import sys
import zmq
#from scipy.sparse import *
import pyximport; pyximport.install()
from PyzmqSampler import *
import subprocess

def compile_models(totalK, models):
    logging.info("Compiling component models into mega-HMM transition and observation matrices")
    (a_max, b_max, g_max) = getVariableMaxes(models)
    
    t0 = time.time()
#    pi = np.zeros((totalK, totalK))
#    phi = np.zeros((totalK, models.lex.dist.shape[1]))
    pi = np.zeros((totalK, totalK))
    phi = np.zeros((totalK, models.lex.dist.shape[1]))
    cache = np.zeros((a_max, b_max, g_max, totalK)) - 1
        
    ## Take exponent out of inner loop:
    word_dist = 10**models.lex.dist
    
    (fork,act,root,cont,start,pos) = unlog_models(models)

    ## Build the observation matrix first:
    for state in range(0,totalK):
        (f,j,a,b,g) = extractStates(state, totalK, getVariableMaxes(models))
        phi[state] = word_dist[g,:]

    ## For previous state i:
    for prevState in range(0,totalK):
        (prevF, prevJ, prevA, prevB, prevG) = extractStates(prevState, totalK, getVariableMaxes(models))
        cumProbs = np.zeros(5)
        
        if cache[prevA][prevB][prevG].sum() > 0:
            pi[prevState] = cache[prevA][prevB][prevG]
        
        else:
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

            cache[prevA][prevB][prevG] = pi[prevState,:]
                            
    time_spent = time.time() - t0
    logging.info("Done in %d s" % time_spent)
#    return (np.matrix(pi, copy=False), np.matrix(phi, copy=False))
    return (np.matrix(pi,copy=False), np.matrix(phi,copy=False))
    
def unlog_models(models):
    fork = 10**models.fork.dist
    act = 10**models.act.dist
    root = 10**models.root.dist
    cont = 10**models.cont.dist
    start = 10**models.start.dist
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


class FiniteSampler(PyzmqSampler):
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, tid, cluster_cmd=None):
        self.cluster_cmd = None
        if cluster_cmd == None:
            PyzmqSampler.__init__(self, host, jobs_port, results_port, models_port, maxLen, tid)
            self.dyn_prog = []
        else:
            cmd_str = 'python3 scripts/finite_sampler.pyx %s %d %d %d %d %d' % (host, jobs_port, results_port, models_port, maxLen, tid)
            self.cluster_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
            #logging.debug("Cluster command is " + str(self.cluster_cmd))
                    
    def read_models(self, models_socket):
        models_socket.send(b'0')
        msg = models_socket.recv_pyobj()
        if msg == None:
            return False
        else:
            (self.models, self.pi, self.phi) = msg
            return True
    
    def initialize_dynprog(self):
        self.dyn_prog = np.zeros((self.K, self.maxLen))
        
    def forward_pass(self,dyn_prog,sent,models,totalK, sent_index):
        ## keep track of forward probs for this sentence:
        maxes = getVariableMaxes(models)
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
                forward[g0_ind+1:g0_ind+g_max-1,0] = np.matrix(10**models.lex.dist[1:-1,token]).transpose()
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

    def reverse_sample(self, dyn_prog, sent, models, totalK, sent_index):            
        sample_seq = []
        sample_log_prob = 0
        maxes = getVariableMaxes(models)
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        ## normalize after multiplying in the transition out probabilities
        dyn_prog[:,last_index] /= dyn_prog[:,last_index].sum()
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                
        sample_seq.append(ihmm.State(extractStates(sample_t, totalK, maxes)))
        if last_index > 0 and (sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0):
            logging.error("Error: First sample has a|b|g = 0")
            sys.exit(-1)
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == 0:
                    continue

                (pf,pj,pa,pb,pg) = extractStates(ind,totalK, getVariableMaxes(models))
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
                    trans_prob *= 10**models.fork.dist[pb, pg, nf]

#                if nf == 0:
#                    trans_prob *= (10**models.reduce.dist[pa,nj])
      
                if nf == 0 and nj == 0:
                    trans_prob *= (10**models.act.dist[pa,na])
                elif nf == 1 and nj == 0:
                    trans_prob *= (10**models.root.dist[pg,na])
                elif nf == 1 and nj == 1:
                    if na != pa:
                        trans_prob = 0
      
                if nj == 0:
                    trans_prob *= (10**models.start.dist[pa,na,nb])
                else:
                    trans_prob *= (10**models.cont.dist[pb,pg,nb])
      
                trans_prob *= (10**models.pos.dist[nb,ng])
                              
                dyn_prog[ind,t] *= trans_prob

            dyn_prog[:,t] /= dyn_prog[:,t].sum()
            sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,t]))
            state_list = extractStates(sample_t, totalK, getVariableMaxes(models))
            
            sample_state = ihmm.State(state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
            sample_seq.append(sample_state)
            
        sample_seq.reverse()
        logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq

def main(args):
    logging.basicConfig(level=logging.INFO)
    fs = FiniteSampler(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5]))
    ## Call run directly instead of start otherwise we'll have 2n workers
    fs.run()
    
if __name__ == "__main__":
    main(sys.argv[1:])
