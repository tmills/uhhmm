#!/usr/bin/env python3.4

import random
import logging
import numpy as np
import ihmm_sampler as sampler

# The set of random variable values at one word
# There will be one of these for every word in the training set
class State:
    def __init__(self):
        self.f = 0
        self.j = 0
        self.a = 0
        self.b = 0
        self.g = 0

    def str(self):
        string = ''
        f_str = '+/' if self.f==1 else '-/'        
        string += f_str
        j_str = '+ ' if self.j==1 else '- '
        string += j_str
        
        string += str(self.a) + '/' + str(self.b) + ':' + str(self.g)
        
        return string

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

    def count(self, cond, out):
        self.condCounts[cond] += 1
        self.pairCounts[cond, out] += 1

    def dec(self, cond, out):
        self.condCounts[cond] -= 1
        self.pairCounts[cond,out] -= 1

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
    sample.beta_a = np.ones((1,start_a+1)) / start_a
    sample.beta_a[0][0] = 0
    sample.beta_b = np.ones((1,start_b+1)) / start_b
    sample.beta_b[0][0] = 0
    sample.beta_g = np.ones((1,start_g+1)) / start_g
    sample.beta_g[0][0] = 0
    sample.beta_f = np.ones((1,2)) / 2
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
        logging.debug("Performing forward pass for iter %d", iter)
        iter += 1
        
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.dist = sampler.sampleDirichlet(models.lex, params['h'])
        models.pos.dist = sampler.sampleDirichlet(models.pos, sample.alpha_g * sample.beta_g)
        models.start.dist = sampler.sampleDirichlet(models.start, sample.alpha_b * sample.beta_b)
        models.cont.dist = sampler.sampleDirichlet(models.cont, sample.alpha_b * sample.beta_b)
        models.act.dist = sampler.sampleDirichlet(models.act, sample.alpha_a * sample.beta_a)
        models.root.dist = sampler.sampleDirichlet(models.root, sample.alpha_a * sample.beta_a)
        models.fork.dist = sampler.sampleDirichlet(models.fork, sample.alpha_f * sample.beta_f)
        a_max = models.act.dist.shape[1]
        b_max = models.cont.dist.shape[1]
        g_max = models.pos.dist.shape[1]
        
        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = 2 * 2 * a_max * b_max * g_max
        dyn_prog = np.zeros((totalK, maxLen+1))
        
        for sent_index,sent in enumerate(ev_seqs):
            decrement_counts(hid_seqs[sent_index], sent, models)
            dyn_prog[:,:] = 0
            ## keep track of forward probs for this sentence:
            for index,token in enumerate(sent):
                if index == 0:
#                    logging.debug("Shape of dynprog is %s", dyn_prog.shape)
#                    logging.debug("Shape of lex model is %s", models.lex.dist.shape)
                    dyn_prog[1:g_max+1,0] = models.lex.dist[:,token] / sum(models.lex.dist[:,token])
                else:
                    for prevInd in np.nonzero(dyn_prog[:,0])[0]:
                        (prevA, prevB, prevG) = extractStates(prevInd, totalK)
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
                            
                            for a in range(0,a_max):
                                    
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
                                
                                for b in range(0,b_max):
                                    if j == 0:
                                        cumProbs[3] = cumProbs[2] * models.cont.dist[prevBG,b]
                                    else:
                                        cumProbs[3] = cumProbs[2] * models.start.dist[prevAa,b]
                                    
                                    for g in range(0,g_max):
                                        cumProbs[4] = cumProbs[3] * models.pos.dist[b,g]
                                        state_index = getStateIndex(f,j,a,b,g)
                                        dyn_prog[state_index,index] = cumProbs[4] * models.lex.dist[g,token]
                                    
        
    return (hid_seqs, stats)


def initialize_state(ev_seqs, models):
    global a_max, b_max, g_max
    a_max = start_a
    b_max = start_b
    g_max = start_g
    
    ## One fork model:
    models.fork = Model(((g_max+1)*(b_max+1), 2))
    ## Two join models:
#    models.trans = Model((g_max*b_max, 2))
#    models.reduce = Model((a_max, 2))
    ## One active model:
    models.act = Model((a_max+1, a_max+1))
    models.root = Model((g_max+1, a_max+1))
    ## two awaited models:
    models.cont = Model(((g_max+1)*(b_max+1),b_max+1))
    models.start = Model(((a_max+1)*(a_max+1), b_max+1))
    ## one pos model:
    models.pos = Model((b_max+1, g_max+1))
    ## one lex model:
    models.lex = Model((g_max+1, max(map(max,ev_seqs))+1))
    
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
                    state.a = random.randint(1,a_max)

                state.b = random.randint(1,b_max)

            state.g = random.randint(1,g_max)
                    
            prev_state = state  
                            
            hid_seq.append(state)
            
        ## special case for end of sentence
        state = State()
        state.f = 0
        state.j = 1
        state.a = 0
        state.b = 0
        state.g = 0
        hid_seq.append(state)
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

            if state.j == 0:
                models.start.count(aa_state(prevState.a, state.a), state.b)
            else:
                models.cont.count(prevBG, state.b)

            
        ## Count G
        models.pos.count(state.b, state.g)
        
        ## Count w
        models.lex.count(state.g, word)
        
        prevState = state
        
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
    f_split = totalK / 2
    if index > f_split:
        index = index - f_split
    
    j_split = f_split / 2
    if index > j_split:
        index = index - j_split
    
    g_ind = index % g_max
    index = (index-g_ind) / g_max
    
    b_ind = index % b_max
    index = (index-b_ind) / b_max
    
    a_ind = index % a_max
    
    return (a_ind, b_ind, g_ind)

def getStateIndex(f,j,a,b,g):
    global a_max, b_max, g_max
    return (((f*2 + j)*a_max + a) * b_max + b)*g_max + g

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
