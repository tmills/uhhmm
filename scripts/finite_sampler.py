#!/usr/bin/env python3

import ihmm
import logging
import time
import numpy as np
import log_math as lm
import sys
from multiprocessing import Process,Queue,JoinableQueue

class FiniteSampler(Sampler):
    def __init__(self, in_q, out_q, models, totalK, maxLen, tid):
        Sampler.__init__(self)
        ## TODO -- compute total model matrix dimensions
        self.state_size = ...
        self.megamodel = compile_model(self.state_size, models)
        
    def compile_model(self, size, models):
        ## TODO -- multiply out model combinations into giant matrix
        ## For previous state i:
        for prevState in range(0,dyn_prog.shape[0]):
            (prevF, prevJ, prevA, prevB, prevG) = ihmm.extractStates(prevState, totalK)
            cumProbs = np.zeros(5)
            prevBG = ihmm.bg_state(prevB,prevG)
            ## Sample f & j:
            for f in (0,1):
                if index == 1 and f == 0:
                    continue

                cumProbs[0] = models.fork.dist[prevBG,f]
                if index == 1:
                    j = 0
                else:
                    j = f
                
                ## At depth 1 -- no probability model for j 
                cumProbs[1] = cumProbs[0]
                
                
                for a in range(1,ihmm.getAmax()):
                    if f == 0 and j == 0:
                        ## active transition:
                        cumProbs[2] = cumProbs[1] * models.act.dist[prevA,a]
                    elif f == 1 and j == 0:
                        ## root -- technically depends on prevA and prevG
                        ## but in depth 1 this case only comes up at start
                        ## of sentence and prevA will always be 0
                        cumProbs[2] = cumProbs[1] * models.root.dist[prevG,a]
                    elif f == 1 and j == 1 and prevA == a:
                        cumProbs[2] = cumProbs[1]
                    else:
                        ## zero probability here
                        continue
                
                    if cumProbs[2] == -np.inf:
                        continue

                    prevAa = ihmm.aa_state(prevA, a)
                
                    for b in range(1,ihmm.getBmax()):
                        if j == 1:
                            cumProbs[3] = cumProbs[2] * models.cont.dist[prevBG,b]
                        else:
                            cumProbs[3] = cumProbs[2] * models.start.dist[prevAa,b]
                    
                        # Multiply all the g's in one pass:
                        ## range gets the range of indices in the forward pass
                        ## that are contiguous in the state space
                        state_range = ihmm.getStateRange(f,j,a,b)
                                             
                        range_probs = cumProbs[3] * models.pos.dist[b,:]
                        
                        dyn_prog[state_range,index] = dyn_prog[state_range,index] + range_probs

                        
    def forward_pass(self,dyn_prog,sent,models,totalK, sent_index):
        ## keep track of forward probs for this sentence:
        ## TODO: Use the giant transition matrix to compute this:
        for index,token in enumerate(sent):
        
            ## Still use special case for 0
            if index == 0:
                g0_ind = ihmm.getStateIndex(0,0,0,0,0)
                dyn_prog[g0_ind:g0_ind+g_max,0] = models.lex.dist[:,token]
                logging.debug(dyn_prog[g0_ind:g0_ind+g_max,0])
            
            else:


                dyn_prog[state_range,index] = lm.log_vector_add(dyn_prog[state_range,index], range_probs)

        
        ## For the last token, multiply in the probability
        ## of transitioning to the end state. also can add up
        ## total probability of data given model here.
        sentence_log_prob = -np.inf
        last_index = len(sent)-1
        for state in range(0,dyn_prog.shape[0]):
            (f,j,a,b,g) = ihmm.extractStates(state, totalK)
            curBG = ihmm.bg_state(b,g)
            dyn_prog[state,last_index] += ((models.fork.dist[curBG,0] + models.reduce.dist[a,1]))
            sentence_log_prob = lm.log_add(sentence_log_prob, dyn_prog[state, last_index])
            logging.debug(dyn_prog[state,last_index])
                       
            if (a == 0 or b == 0 or g == 0) and dyn_prog[state, last_index] != -np.inf:
                logging.error("Error: Non-zero probability at g=0 in forward pass!")
                sys.exit(-1)

        if np.argwhere(dyn_prog.max(0)[0:last_index+1] == -np.inf).size > 0:
            logging.error("Error; There is a word with no positive probabilities for its generation")
            sys.exit(-1)

        return dyn_prog, sentence_log_prob

    def reverse_sample(self, dyn_prog, sent, models, totalK, sent_index):            
        sample_seq = []
        sample_log_prob = 0
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        dyn_prog[:,last_index] = lm.normalize_from_log(dyn_prog[:,last_index])
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                
        sample_seq.append(ihmm.State(ihmm.extractStates(sample_t, totalK)))
        if sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0:
            logging.error("Error: First sample has a|b|g = 0")
            sys.exit(-1)
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == -np.inf:
                    continue

                (pf,pj,pa,pb,pg) = ihmm.extractStates(ind,totalK)
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                prevBG = ihmm.bg_state(pb,pg)
                trans_prob = models.fork.dist[prevBG,nf]
                if nf == 0:
                    trans_prob += models.reduce.dist[pa,nj]
      
                if nf == 0 and nj == 0:
                    trans_prob += models.act.dist[pa,na]
                elif nf == 1 and nj == 0:
                    trans_prob += models.root.dist[pg,na]
                elif nf == 1 and nj == 1:
                    if na != pa:
                        trans_prob = -np.inf
      
                if nj == 0:
                    prevAA = ihmm.aa_state(pa,na)
                    trans_prob += models.start.dist[prevAA,nb]
                else:
                    trans_prob += models.cont.dist[prevBG,nb]
      
                trans_prob += models.pos.dist[nb,ng]
                if np.isnan(trans_prob):
                    logging.error("pos model is nan!")
                    
                dyn_prog[ind,t] += trans_prob

            normalized_dist = lm.normalize_from_log(dyn_prog[:,t])
            sample_t = sum(np.random.random() > np.cumsum(normalized_dist))
            state_list = ihmm.extractStates(sample_t, totalK)
            
            sample_state = ihmm.State(state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
            sample_seq.append(sample_state)
            dyn_prog[:,t] = normalized_dist
            
        sample_seq.reverse()
        logging.debug("Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq
