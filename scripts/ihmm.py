#!/usr/bin/env python3.4

import random
import logging
import time
import numpy as np
import ihmm_sampler as sampler
import pdb
from threading import Thread
from multiprocessing import Process

# The set of random variable values at one word
# There will be one of these for every word in the training set
class State:
    def __init__(self, state=None):
        if state == None:
            self.f = 0
            self.j = 0
            self.a = 0
            self.b = 0
            self.g = 0
        else:
            (self.f, self.j, self.a, self.b, self.g) = state

    def str(self):
        string = ''
        f_str = '+/' if self.f==1 else '-/'        
        string += f_str
        j_str = '+ ' if self.j==1 else '- '
        string += j_str
        
        string += str(self.a) + '/' + str(self.b) + ':' + str(self.g)
        
        return string

    def to_list(self):
        return (self.f, self.j, self.a, self.b, self.g)

# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seq = []
        

# Historgam of how many instances of each state you have random sampled
# May be a field in Sample
class Stats:
    def __init__(self):
        ## number of each type of variable:
        self.numA = 0
        self.numB = 0
        self.numG = 0
        ## alpha is concentration parameter to dirichlet process/pitman yor model
        self.alpha0 = 0
        self.gamma = 0
        self.vi = 0


class Model:
    def __init__(self, shape):
        self.condCounts = np.zeros((shape[0],1), dtype=np.uint)
        self.pairCounts = np.zeros(shape, dtype=np.uint)
        self.dist = None

    def count(self, cond, out):
        self.condCounts[cond] += 1
        self.pairCounts[cond, out] += 1

    def dec(self, cond, out):
        self.condCounts[cond] -= 1
        self.pairCounts[cond,out] -= 1

    def sampleDirichlet(self, base):
        nullState = True
        self.dist = sampler.sampleDirichlet(self, base, nullState)
        self.condCounts[:] = 0
        self.pairCounts[:] = 0

    def sampleBernoulli(self, base):
        self.dist = sampler.sampleDirichlet(self, base)
        self.condCounts[:] = 0
        self.pairCounts[:] = 0

class Models(list):
    def resample_all():
        for model in self:
            model.dist = sampleDirichlet(model)

def sample_beam(ev_seqs, params):    
    
    global start_a, start_b, start_g 
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))

    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    samples = int(params.get('num_samples'))
    num_iters = burnin + (samples-1)*iters
    maxLen = max(map(len, ev_seqs))
    
    sample = Sample()
    sample.alpha_a = float(params.get('alphaa'))
    sample.alpha_b = float(params.get('alphab'))
    sample.alpha_g = float(params.get('alphag'))
    sample.alpha_f = float(params.get('alphaf'))
    sample.alpha_j = float(params.get('alphaj'))
    sample.beta_a = np.ones((1,start_a+1)) / start_a
    sample.beta_a[0][0] = 0
    sample.beta_b = np.ones((1,start_b+1)) / start_b
    sample.beta_b[0][0] = 0
    sample.beta_g = np.ones((1,start_g+1)) / start_g
    sample.beta_g[0][0] = 0
    sample.beta_f = np.ones((1,2)) / 2
    sample.beta_j = np.ones((1,2)) / 2
    sample.gamma = float(params.get('gamma'))
    sample.discount = float(params.get('discount'))
    
    
    models = Models()
    
    logging.info("Initializing state")    
    hid_seqs = initialize_state(ev_seqs, models)
    sample.S = hid_seqs
    
    stats = Stats()
    
    logging.debug(ev_seqs[0])
    logging.debug(list(map(lambda x: x.str(), hid_seqs[0])))
    
    iter = 0
    while iter < num_iters:
        iter += 1
        
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.sampleDirichlet(params['h'])
        models.pos.sampleDirichlet(sample.alpha_g * sample.beta_g)
        models.start.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.cont.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.act.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.root.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.reduce.sampleBernoulli(sample.alpha_j * sample.beta_j)
        models.fork.sampleBernoulli(sample.alpha_f * sample.beta_f)
        ## These values keep track of actual maxes not user-specified --
        ## so if user specifies 10 to start this will be 11 because of state 0 (init)
        a_max = models.act.dist.shape[1]
        b_max = models.cont.dist.shape[1]
        g_max = models.pos.dist.shape[1]
        
        sample = Sample()
        sample.S = []
        
        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = 2 * 2 * a_max * b_max * g_max
        inf_procs = dict()
        t0 = time.time()
        num_procs = 4
        cur_proc = 0
        dyn_prog = np.zeros((num_procs,totalK, maxLen+1))

        
        for sent_index,sent in enumerate(ev_seqs):
            if sent_index > 0 and sent_index % 100 == 0:
                t1 = time.time()
                logging.info("Iteration %d, sentence %d: Spent %d s on last 100 sentences", iter, sent_index, t1-t0)
                t0 = time.time()

            if len(inf_procs) == num_procs:
                #logging.debug("Waiting for thread %d to join", cur_thread)
                inf_procs[cur_proc].join()
                sent_sample = inf_procs[cur_proc].get_sample()
                sample.S.append(sent_sample)


            #logging.debug("Spawning thread number %d for sent index %d", cur_thread, sent_index)

            sampler_proc = Sampler(dyn_prog[cur_proc,:,:], models, totalK, maxLen+1, cur_proc)
            inf_procs[cur_proc] = sampler_proc
            sampler_proc.set_data(sent)
            sampler_proc.start()
            cur_proc = (cur_proc+1) % num_procs
            
    return (hid_seqs, stats)

class Sampler(Process):
    def __init__(self, dyn_prog, models, totalK, maxLen, tid):
        Process.__init__(self)
        self.models = models
        self.K = totalK
        self.sent_sample = None
        self.sent = None
        self.dyn_prog = dyn_prog
        self.tid = tid
    
    def set_data(self, sent):
        self.sent = sent

    def run(self):
        self.dyn_prog[:,:] = 0
        t0 = time.time()
        #logging.debug("Starting forward pass in thread %s", self.tid)
        self.dyn_prog = self.forward_pass(self.dyn_prog, self.sent, self.models, self.K)
        #logging.debug("Starting backwards pass in thread %s", self.tid)
        self.sent_sample = self.reverse_sample(self.dyn_prog, self.sent, self.models, self.K)
        increment_counts(self.sent_sample, self.sent, self.models)
        t1 = time.time()
        #logging.debug("Thread %d required %d s to process sentence.", self.tid, (t1-t0))

    def get_sample(self):
        return self.sent_sample
    
    def forward_pass(self,dyn_prog,sent,models,totalK):
        ## keep track of forward probs for this sentence:
        for index,token in enumerate(sent):
            if index == 0:
                g0_ind = getStateIndex(0,0,0,0,0)
                dyn_prog[g0_ind:g0_ind+g_max,0] = models.lex.dist[:,token] / sum(models.lex.dist[:,token])
            else:
                for prevInd in np.nonzero(dyn_prog[:,index-1])[0]:
                    (prevF, prevJ, prevA, prevB, prevG) = extractStates(prevInd, totalK)

                    assert index == 1 or (prevA != 0 and prevB != 0 and prevG != 0), 'Unexpected values in sentence {0} with non-zero probabilities: {1}, {2}, {3} at index {4}, and f={5} and j={6}'.format(sent_index,prevA, prevB, prevG, index, prevF, prevJ)
                
                    cumProbs = np.zeros((5,1))
                    prevBG = bg_state(prevB,prevG)
                
                    ## Sample f & j:
                    for f in (0,1):
                        cumProbs[0] = models.fork.dist[prevBG,f]
                        if index == 1:
                            j = 0
                        else:
                            j = f
                    
                        cumProbs[1] = cumProbs[0]
                    
                        for a in range(1,a_max):
    #                                pdb.set_trace()
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
                        
                            prevAa = aa_state(prevA, a)
                        
                            for b in range(1,b_max):
                                if j == 0:
                                    cumProbs[3] = cumProbs[2] * models.cont.dist[prevBG,b]
                                else:
                                    cumProbs[3] = cumProbs[2] * models.start.dist[prevAa,b]
                            
                                # Multiply all the g's in one pass:
                                state_range = getStateRange(f,j,a,b)
                                dyn_prog[state_range,index] = cumProbs[3] * models.pos.dist[b,:] * models.lex.dist[:,token]

                                ## For the last token, we can multiply in the
                                ## probability of ending the sentence right away:
                                if index == len(sent)-1:
                                    for g in range(1,g_max):
                                        curBG = bg_state(b,g)
                                        dyn_prog[state_range[0]+g,index] *= (models.fork.dist[curBG,0] * models.reduce.dist[a,1])
    
            ## Normalize at this time step:
            dyn_prog[:, index] /= sum(dyn_prog[:,index])
        return dyn_prog

    def reverse_sample(self,dyn_prog,sent,models,totalK):            
        sample_seq = []
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,len(sent)-1]))
        sample_seq.append(State(extractStates(sample_t, totalK)))
      #            if sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0:
      #                pdb.set_trace()
  
        for t in range(len(sent)-2,-1,-1):
            for ind in np.nonzero(dyn_prog[:,t])[0]:
                (pf,pj,pa,pb,pg) = extractStates(ind,totalK)
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                prevBG = bg_state(pb,pg)
                trans_prob = models.fork.dist[prevBG,nf]
                if nf == 0:
                    trans_prob *= models.reduce.dist[pa,nj]
      
                if nf == 0 and nj == 0:
                    trans_prob *= models.act.dist[pa,na]
                elif nf == 1 and nj == 0:
                    trans_prob *= models.root.dist[pg,na]
      
      
                if nj == 0:
                    prevAA = aa_state(pa,na)
                    trans_prob *= models.start.dist[prevAA,nb]
                else:
                    trans_prob *= models.cont.dist[prevBG,nb]
      
                trans_prob *= models.pos.dist[nb,ng]
                dyn_prog[ind,t] *= trans_prob

      #                if sum(dyn_prog[:,t]) == 0.0:
      #                    pdb.set_trace()
            dyn_prog[:,t] /= sum(dyn_prog[:,t])
            sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,t]))
            sample_seq.append(State(extractStates(sample_t, totalK)))

        sample_seq.reverse()
      #            logging.debug("Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq


def initialize_state(ev_seqs, models):
    global a_max, b_max, g_max
    a_max = start_a+1
    b_max = start_b+1
    g_max = start_g+1
    
    ## One fork model:
    models.fork = Model(((g_max)*(b_max), 2))
    ## Two join models:
#    models.trans = Model((g_max*b_max, 2))
    models.reduce = Model((a_max, 2))
    ## One active model:
    models.act = Model((a_max, a_max))
    models.root = Model((g_max, a_max))
    ## two awaited models:
    models.cont = Model(((g_max)*(b_max),b_max))
    models.start = Model(((a_max)*(a_max), b_max))
    ## one pos model:
    models.pos = Model((b_max, g_max))
    ## one lex model:
    models.lex = Model((g_max, max(map(max,ev_seqs))+1))
    
    logger.debug("Value of amax=%d, b_max=%d, g_max=%d", a_max, b_max, g_max)
    
    state_seqs = list()
    for sent in ev_seqs:
        hid_seq = list()
        for index,word in enumerate(sent):
            state = State()
            ## special case for first word
            if index == 0:
                state.f = 0
                state.j = 0
                state.a = 0
                state.b = 0
            else:
                if index == 1:
                    state.f = 1
                    state.j = 0
                else:
                    if random.random() > 0.5:
                        state.f = 1
                    else:
                        state.f = 0
                    ## j is deterministic in the middle of the sentence
                    state.j = state.f
                    
                if state.f == 1 and state.j == 1:
                    state.a = prev_state.a
                else:
                    state.a = np.random.randint(1,a_max)

                state.b = np.random.randint(1,b_max)

            state.g = np.random.randint(1,g_max)
                    
            prev_state = state  
                            
            hid_seq.append(state)
            
        ## special case for end of sentence
#        state = State()
#        state.f = 0
#        state.j = 1
#        state.a = 0
#        state.b = 0
#        state.g = 0
#        hid_seq.append(state)
        increment_counts(hid_seq, sent, models)
        state_seqs.append(hid_seq)

    return state_seqs

def increment_counts(hid_seq, sent, models):
    ## for every state transition in the sentence increment the count
    ## for the condition and for the output
    for index,word in enumerate(sent):
        state = hid_seq[index]
        if index != 0:
            prevBG = bg_state(prevState.b, prevState.g)
            ## Count F & J
            if index == 1:
                models.root.count(prevState.g, state.a)
            else:
                models.fork.count(prevBG, state.f)

                ## Count A & B
                if state.f == 0 and state.j == 0:
                    models.act.count(prevState.a, state.a)

            if state.f == 0 and state.j == 0:
                models.reduce.count(prevState.a, state.j)
                                
            if state.j == 0:
                models.start.count(aa_state(prevState.a, state.a), state.b)
            else:
                models.cont.count(prevBG, state.b)

            
        ## Count G
        models.pos.count(state.b, state.g)
        
        ## Count w
        models.lex.count(state.g, word)
        
        prevState = state
    
    prevBG = bg_state(hid_seq[-1].b, hid_seq[-1].g)
    models.fork.count(prevBG, 0)
    models.reduce.count(hid_seq[-1].a, 1)

## Don't think we actually need this since we are not using counts to integrate
## out models -- instead we sample models so we can just reset all counts to 0
## after resampling models at each iteration
def decrement_counts(hid_seq, sent, models):
    ## for every state transition in the sentence decrement the count
    ## for the condition and and the generated value
    for index,word in enumerate(sent):
        state = hid_seq[index]
        if index != 0:
            prevBG = bg_state(prevState.b, prevState.g)
            if index == 1:
                models.root.dec(prevState.g, state.a)
            else:
                
                models.fork.dec(prevBG, state.f)
                
                if state.f == 0 and state.j == 0:
                    models.act.dec(prevState.a, state.a)
                
            if state.j == 0:
                models.start.dec(aa_state(prevState.a, state.a), state.b)
            else:
                models.cont.dec(prevBG, state.b)
                
        models.pos.dec(state.b, state.g)
        models.lex.dec(state.g, sent[index])
        
        prevState = state


def bg_state(b, g):
    global g_max
    return b*g_max + g

def aa_state(prevA, a):
    global a_max
    return prevA * a_max + a
    
def extractStates(index, totalK):
    global a_max, b_max, g_max
    
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
    
    return (f_ind,j_ind,a_ind, b_ind, g_ind)

def getStateIndex(f,j,a,b,g):
    global a_max, b_max, g_max
    return (((f*2 + j)*a_max + a) * b_max + b)*g_max + g

def getStateRange(f,j,a,b):
    global g_max
    start = getStateIndex(f,j,a,b,0)
    return range(start,start+g_max)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
