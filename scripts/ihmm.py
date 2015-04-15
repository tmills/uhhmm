#!/usr/bin/env python3.4

import random
import logging
import numpy as np
import ihmm_sampler as sampler

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

class Sample:
    def __init__(self):
        self.hid_seq = []
        
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

class Models():
    pass

def sample_beam(ev_seqs, params):    
    
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    samples = int(params.get('num_samples'))
    num_iters = burnin + (samples-1)*iters
    
    sample = Sample()
    sample.alpha0 = params.get('alpha0')
    sample.gamma = params.get('gamma')
    sample.discount = params.get('discount')
    
    global start_a, start_b, start_g 
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))
    
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

        ## TODO -- grab this from thee sample sequences
        g_max = start_g
        
        ## Sample distributions for all the model params and emissions params
        sample.Phi = sampler.sampleEmissionMatrix(sample.S, ev_seqs, g_max, params.H);
        
        
    return (hid_seqs, stats)


def initialize_state(ev_seqs, models):
    a_max = start_a
    b_max = start_b
    g_max = start_g
    
    ## One fork model:
    f_model = Model(g_max*b_max, 2)
    ## Two join models:
    trans_model = Model(g_max*b_max, 2)
    red_model = Model(a_max, 2)
    ## One active model:
    act_model = Model(a_max, a_max)
    ## two awaited models:
    cont_model = Model(g_max*b_max,b_max)
    start_model = Model(a_max*a_max, b_max)
    ## one pos model:
    g_model = Model(b_max, g_max)
    ## one lex model:
    w_model = Model(g_max, ev_seqs.max())
    
    models.fork = f_model
    models.trans = trans_model
    models.reduce = red_model
    models.act = act_model
    models.cont = cont_model
    models.start = start_model
    models.pos = g_model
    models.lex = w_model
    
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
        if index == len(sent)-1:
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
    for index,state in hid_seq:
        if index != 0:
            ## Count F & J
            models.fork.count(prevState.b, prevState.g, state.f)
            if state.f == 0
                models.reduce.count(prevState.a, state.j)
            else:
                models.reduce.count(prevState.b, prevState.g, state.j)

            ## Count A & B
            if state.f == 0 and state.j == 0:
                models.act.count(prevState.a, state.a)
                
            if state.j == 0:
                models.start.count(prevState.a, state.a, state.b)
            else 
                models.cont.count(prevState.b, prevState.g, state.b)

            
        ## Count G
        models.pos.count(state.b, state.g)
        
        ## Count w
        models.lex.count(state.g, sent(index))
        
            
def decrement_counts(hid_seq, sent, models):
    ## for every state transition in the sentence decrement the count
    ## for the condition and and the generated value
    print("Not implemented yet")
    
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
