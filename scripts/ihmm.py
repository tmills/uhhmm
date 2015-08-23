#!/usr/bin/env python3.4

import random
import logging
import time
import numpy as np
import ihmm_sampler as sampler
import ihmm_io
import pdb
import sys
from log_math import *
from multiprocessing import Process,Queue,JoinableQueue

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
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0

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

# A mapping from input space to output space. The Model class can be
# used to store counts during inference, and then know how to resample themselves
# if given a base distribution.
# TODO: Sub-class for BooleanModel vs. InfiniteModel  with single sample()/resample() method
# and automatically adjusting sizes for infinite version.
class Model:
    def __init__(self, shape, alpha=0.0, beta=None, corpus_shape=(0,0)):
        self.pairCounts = np.zeros(shape, dtype=np.uint)
        self.dist = np.random.random(shape)
        self.dist /= self.dist.sum(1, keepdims=True)
        self.dist = np.log10(self.dist)
        self.u = np.zeros(corpus_shape) + -np.inf
        self.trans_prob = np.zeros(corpus_shape)
        self.alpha = alpha
        self.beta = beta

    def count(self, cond, out):
        self.pairCounts[cond, out] += 1

    def dec(self, cond, out):
        self.pairCounts[cond,out] -= 1

    def selfSampleDirichlet(self):
        self.sampleDirichlet(self.alpha * self.beta)
        
    def sampleDirichlet(self, base):
        self.dist = sampler.sampleDirichlet(self, base)
        self.pairCounts[:,:] = 0

    def sampleBernoulli(self, base):
        self.dist = sampler.sampleBernoulli(self, base)
        self.pairCounts[:,:] = 0

# This class is not currently used. Could someday be used to resample
# all models if we give Model s more information about themselves.
class Models(list):
    def resample_all():
        for model in self:
            model.dist = sampleDirichlet(model)

# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
def sample_beam(ev_seqs, params, report_function, pickle_file=None):    
    
    global start_a, start_b, start_g
    global a_max, b_max, g_max
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting beam sampling")
    
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    num_samples = int(params.get('num_samples'))
    num_procs = int(params.get('num_procs'))
    debug = bool(int(params.get('debug', 0)))
    profile = bool(int(params.get('profile', 0)))
    finite = bool(int(params.get('finite', 0)))
    
    if not profile:
        logging.info('profile is set to %s, importing and installing pyx' % profile)    
        import pyximport; pyximport.install()

    import beam_sampler
    import finite_sampler

    samples = []
    
    maxLen = max(map(len, ev_seqs))
    max_output = max(map(max, ev_seqs))
    
    models = Models()
    
    logging.info("Initializing state")
    
    if pickle_file == None:
        ## Add 1 to every start value for "Null/start" state
        a_max = start_a+2
        b_max = start_b+2
        g_max = start_g+2

        models = initialize_models(models, max_output, params, (len(ev_seqs), maxLen))
        hid_seqs = initialize_state(ev_seqs, models)

        sample = Sample()
    #    sample.hid_seqs = hid_seqs
        sample.alpha_a = models.root.alpha ## float(params.get('alphaa'))
        sample.alpha_b = float(params.get('alphab'))
        sample.alpha_g = float(params.get('alphag'))
        sample.alpha_f = float(params.get('alphaf'))
        sample.alpha_j = float(params.get('alphaj'))
        ## use plus 2 here (until moved later) since we need the null state (0) as well as 
        ## the extra part of the stick for "new tables"
        sample.beta_a = models.root.beta ## np.ones((1,start_a+2)) / start_a
    #    sample.beta_a[0][0] = 0
        sample.beta_b = models.cont.beta
    #    sample.beta_b[0] = 0
        sample.beta_g = models.pos.beta
        sample.beta_f = np.ones(2) / 2
        sample.beta_j = np.ones(2) / 2
        sample.gamma = float(params.get('gamma'))
        sample.discount = float(params.get('discount'))
    
    
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.sampleDirichlet(params['h'])
        models.pos.selfSampleDirichlet()
        if np.argwhere(np.isnan(models.pos.dist)).size > 0:
            logging.error("Resampling the pos dist resulted in a nan")
        
        models.start.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.cont.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.act.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.root.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.reduce.sampleBernoulli(sample.alpha_j * sample.beta_j)
        models.fork.sampleBernoulli(sample.alpha_f * sample.beta_f)
    
        sample.models = models
        iter = 0

    else:
        sample = ihmm_io.read_serialized_sample(pickle_file)
        models = sample.models
        hid_seqs = sample.hid_seqs
        sample.hid_seqs = [] ## Empty out hid_seqs because we will append later.
        
        a_max = models.act.dist.shape[1]
        b_max = models.cont.dist.shape[1]
        g_max = models.pos.dist.shape[1]
        
        iter = sample.iter+1

    collect_trans_probs(hid_seqs, models)
    
    stats = Stats()
    
    logging.debug(ev_seqs[0])
    logging.debug(list(map(lambda x: x.str(), hid_seqs[0])))
    
    
    while len(samples) < num_samples:
        sample.iter = iter

        
        if not finite:
            models.pos.u =  models.pos.trans_prob +  np.log10(np.random.random((len(ev_seqs), maxLen)) )
            models.root.u = models.root.trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )
            models.cont.u = models.cont.trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )
        
            ## Break off the beta sticks before actual processing -- instead of determining during
            ## inference whether to create a new state we use an auxiliary variable u to do it
            ## ahead of time, but note there is no guarantee we will ever use it.
            ## TODO: Resample beta, which will allow for unused probability mass to go to the end again?
            while a_max < 20 and models.root.u.min() < max(models.root.dist[:,-1].max(),models.act.dist[:,-1].max()):
                logging.info('Breaking a stick')
                break_a_stick(models, sample, params)
            
            if a_max >= 20:
                logging.warn('Stick-breaking (a) terminated due to hard limit and not gracefully.')
        
            while b_max < 50 and models.cont.u.min() < models.cont.dist[:,-1].max():
                logging.info('Breaking b stick')
                break_b_stick(models, sample, params)

            if b_max >= 50:
                logging.warn('Stick-breaking (b) terminated due to hard limit and not gracefully.')

            while g_max < 50 and models.pos.u.min() < models.pos.dist[:,-1].max():
                logging.info('Breaking g stick')
                break_g_stick(models, sample, params)
                if np.argwhere(np.isnan(models.pos.dist)).size > 0:
                    logging.error("Breaking the g stick resulted in a nan in the output distribution")
            
            if g_max >= 50:
                logging.warn('Stick-breaking (g) terminated due to hard limit and not gracefully.')
            
        ## These values keep track of actual maxes not user-specified --
        ## so if user specifies 10 to start this will be 11 because of state 0 (init)
        a_max = models.act.dist.shape[1]
        b_max = models.cont.dist.shape[1]
        g_max = models.pos.dist.shape[1]
        
        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = 2 * 2 * a_max * b_max * g_max

        logging.info("Number of a states=%d, b states=%d, g states=%d, total=%d" % (a_max-2, b_max-2, g_max-2, totalK))
        
        
        if finite:
            (trans_mat, obs_mat) = finite_sampler.compile_models(totalK, models)
        
        inf_procs = dict()
        cur_proc = 0

        sent_q = JoinableQueue() ## Input queue
        state_q = Queue() ## Output queue
        
        
        logging.info("Placing sentences into shared queue")
        
        ## Place all sentences into the input queue
        for sent_index,sent in enumerate(ev_seqs):
            sent_q.put((sent_index,sent))
            
        ## Initialize all the sub-processes with their input-output queues,
        ## read-only models, and dimensions of matrix they'll need
        t0 = time.time()
            
        for cur_proc in range(0,num_procs):
            ## For each sub-process add a "None" element to the queue that tells it that
            ## we are out of sentences (we've added them all above)
            sent_q.put(None)
            
            ## Initialize and start the sub-process
            if finite:
                inf_procs[cur_proc] = finite_sampler.FiniteSampler(trans_mat, obs_mat, sent_q, state_q, models, totalK, maxLen+1, cur_proc)
            else:
                inf_procs[cur_proc] = beam_sampler.Sampler(sent_q, state_q, models, totalK, maxLen+1, cur_proc)
            if debug:
                ## calling run instead of start just treats it like a plain object --
                ## doesn't actually do a fork. So we'll just block here for it to finish
                ## rather than needing to join later.
                ## Then we can use pdb() for debugging inside the thread.
                inf_procs[cur_proc].run()
            else:
                inf_procs[cur_proc].start()

        ## Close the queue
        sent_q.join()
        t1 = time.time()
        logging.info("Sampling time for iteration %d is %d s" % (iter, t1-t0))
        
        t0 = time.time()
        num_processed = 0
        sample_map = dict()
        while not state_q.empty():
            num_processed += 1
            (sent_index, sent_sample, log_prob) = state_q.get()
            #logging.debug("Incrementing count for sent index %d and %d sentences left in queue" % (sent_index, len(ev_seqs)-num_processed))
#            if sent_index % 10 == 0:
#                logging.info("Processed sentence {0}".format(sent_index))
            #pdb.set_trace()
            increment_counts(sent_sample, ev_seqs[sent_index], models, sent_index)
            sample_map[sent_index] = sent_sample
            sample.log_prob += log_prob

        ## samples got unsorted by queueing them so resort them just for the purpose 
        ## of debugging.
        for key in sorted(sample_map.keys()):
            sample.hid_seqs.append(sample_map[key])

        t1 = time.time()
        logging.info("Building counts tables took %d s" % (t1-t0))
        
        if iter >= burnin and (iter-burnin) % iters == 0:
            samples.append(sample)
            report_function(sample)
            logging.info(".\n")
        
        t0 = time.time()

        next_sample = Sample()
        

        
        ## Sample hyper-parameters
        ## This is, e.g., where we might add categories to the a,b,g variables with
        ## stick-breaking. Without that, the values will stay what they were 
        next_sample.alpha_f = sample.alpha_f
        next_sample.beta_f = sample.beta_f
        next_sample.alpha_j = sample.alpha_j
        next_sample.beta_j = sample.beta_j
        
        next_sample.alpha_a = models.root.alpha
        next_sample.beta_a = models.root.beta
        next_sample.alpha_b = models.cont.alpha
        next_sample.beta_b = models.cont.beta
#        next_sample.alpha_g = sample.alpha_g
#        next_sample.beta_g = sample.beta_g
        next_sample.alpha_g = models.pos.alpha
        next_sample.beta_g = models.pos.beta
        next_sample.gamma = sample.gamma
        
        prev_sample = sample
        sample = next_sample

        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.sampleDirichlet(params['h'])
        models.pos.selfSampleDirichlet()
        models.start.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.cont.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.act.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.root.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.reduce.sampleBernoulli(sample.alpha_j * sample.beta_j)
        models.fork.sampleBernoulli(sample.alpha_f * sample.beta_f)
        t1 = time.time()
        
        logging.debug("Resampling models took %d s" % (t1-t0))
        
        ## now that we've resampled models, store the transition probabilities that
        ## the new model assigned to all the transitions
        collect_trans_probs(prev_sample.hid_seqs, models)
        
        sample.models = models
        
        iter += 1
        

    return (samples, stats)

def add_model_column(model):
    num_conds = model.dist.shape[0]
    model.pairCounts = np.append(model.pairCounts, np.zeros((num_conds, 1)), 1)
    dist_end = model.dist[:,-1]
    param_a = np.tile(model.alpha * model.beta[-2], (num_conds, 1))
    param_b = model.alpha * (1 - model.beta[0:-1].sum())
    if param_a.min() < 1e-2 or param_b < 1e-2:
        pg = np.random.binomial(1, param_a / (param_a+param_b)).flatten()
    else:
        pg = np.random.beta(param_a, param_b).flatten()

    model.dist[:,-1] = np.log10(pg)
    model.dist = np.append(model.dist, np.tile(np.log10(1-pg) + dist_end, (1,1)).transpose(), 1)
    if np.argwhere(np.isnan(model.dist)).size > 0:
        logging.error("Addition of column resulted in nan!")

def add_model_row_simple(model, base):
    num_outs = model.dist.shape[1]
    model.pairCounts = np.append(model.pairCounts, np.zeros((1,num_outs)), 0)
    model.dist = np.append(model.dist, np.zeros((1,num_outs)), 0)
    model.dist[-1,0] = -np.inf
    model.dist[-1,1:] = np.log10(sampler.sampleSimpleDirichlet(model.pairCounts[-1,1:] + base))
    if np.argwhere(np.isnan(model.dist)).size > 0:
        logging.error("Addition of column resulted in nan!")

def break_beta_stick(model, gamma):
    beta_end = model.beta[-1]
    new_group_fraction = np.random.beta(1, gamma)
    model.beta = np.append(model.beta, np.zeros(1))
    model.beta[-2] = new_group_fraction * beta_end
    model.beta[-1] = (1-new_group_fraction) * beta_end

def break_a_stick(models, sample, params):
    global a_max, b_max, g_max
    
    a_max += 1
    
    ## Break the a stick (stored in root by convention -- TODO --  move this out to its own class later)
    break_beta_stick(models.root, sample.gamma)
    models.act.beta = models.root.beta
    
    ## Add a column to each of the out distributions (ACT and ROOT)
    add_model_column(models.root)
    add_model_column(models.act)  
    
    ## Add a row to the j distribution (TODO)
    ## Add a row to the ACT distributions (which depends on a_{t-1})
    add_model_row_simple(models.act, models.act.alpha * models.act.beta[1:])
    
    ## For boolean variables can't do the simple row add:
    models.reduce.pairCounts = np.append(models.reduce.pairCounts, np.zeros((1,2)), 0)
    new_dist = np.log10([[0.5, 0.5]])
    models.reduce.dist = np.append(models.reduce.dist, new_dist, 0)
    
    old_start = models.start.pairCounts
    models.start.pairCounts = np.zeros((a_max*a_max,b_max))
    old_start_dist = models.start.dist
    models.start.dist = np.zeros((a_max*a_max,b_max))
    old_start_ind = 0
    
    ## Add intermittent rows to the start distribution (depends on a_{t-1}, a_t)
    ## Special case -- because both variables are 'a', we can't go all the way to a_max in the
    ## range variable -- that will take us too far. The last case we will handle just below
    ## this loop and do all at once.
    for a in range(0,a_max-1):
        aa = a * a_max
        models.start.pairCounts[aa:aa+a_max-1,:] = old_start[old_start_ind:old_start_ind+a_max-1,:]
        models.start.dist[aa:aa+a_max-1,:] = old_start_dist[old_start_ind:old_start_ind+a_max-1,:]
        models.start.dist[aa+a_max-1,0] = -np.inf
        models.start.dist[aa+a_max-1,1:] = np.log10(sampler.sampleSimpleDirichlet(sample.alpha_b * sample.beta_b[1:]))
        old_start_ind += a_max - 1

    ## Also need to add a whole block at the end
    aa = a_max * (a_max - 1)
    for a in range(0,a_max):
        models.start.dist[aa+a,0] = -np.inf
        models.start.dist[aa+a,1:] = np.log10(sampler.sampleSimpleDirichlet(sample.alpha_b * sample.beta_b[1:]))

def break_b_stick(models, sample, params):
    global a_max, b_max, g_max
    
    b_max += 1
    
    ## Keep the stick with cont and copy it over to beta
    break_beta_stick(models.cont, sample.gamma)
    models.start.beta = models.cont.beta
    
    ## Add a column to both output distributions:
    add_model_column(models.cont)
    add_model_column(models.start)
    
    ## Add a row to the POS output distribution which depends only on b:
    add_model_row_simple(models.pos, sample.alpha_g * sample.beta_g[1:])
    
    ## Several models depend on both b & g: Fork (boolean), Trans (boolean), Cont (awaited).
    ## Since g is the "inside" variable, when we increase b we just add a block of distributions
    ## the size of g to the end (in contrast, when we break the g stick [just below] 
    ## we intermittently add rows)
    models.cont.dist = np.append(models.cont.dist, np.zeros((g_max, models.cont.dist.shape[1])), 0)
    models.cont.pairCounts = np.append(models.cont.pairCounts, np.zeros((g_max, models.cont.pairCounts.shape[1])), 0)
    models.fork.dist = np.append(models.fork.dist, np.zeros((g_max, 2)), 0)
    models.fork.pairCounts = np.append(models.fork.pairCounts, np.zeros((g_max,2)), 0)
    
    bg = (b_max-1) * g_max
    for g in range(0, g_max):
        new_cont = np.log10(sampler.sampleSimpleDirichlet(models.cont.alpha * models.cont.beta[1:]))
        models.cont.dist[bg + g,0] = -np.inf
        models.cont.dist[bg + g,1:] = new_cont
    
        models.fork.dist[bg + g,:] = np.log10(sampler.sampleSimpleBernoulli(sample.alpha_f * sample.beta_f))
    
    
def break_g_stick(models, sample, params):
    global a_max, b_max, g_max
    
    g_max += 1
    num_conds = models.pos.dist.shape[0]

    ## Resample beta when the stick is broken:
    break_beta_stick(models.pos, sample.gamma)
    
    if models.pos.beta[-1] == 0.0:
        logging.error("This shouldn't be 0!")
    
    ## Add a column to the distribution that outputs POS tags:
    add_model_column(models.pos)

    ## Add a row to the lexical distribution for this new POS tag:
    add_model_row_simple(models.lex, params['h'][0,1:])
    
    ## Add a row to the active (a) model for the new conditional value of g 
    add_model_row_simple(models.root, models.root.alpha * models.root.beta[1:])
    
    ## The slightly trickier case of distributions which depend on g as well as
    ## other variables (in this case, both depend on b) : Need to grab out slices of 
    ## distributions and insert into new model with gaps in interior rows

    ## Add rows to the input distributions for all the models dependent on g
    ## at the next time step: (trans [not used yet], cont)
    old_cont = models.cont.pairCounts
    models.cont.pairCounts = np.zeros((b_max*g_max,b_max))
    old_cont_dist = models.cont.dist
    models.cont.dist = np.zeros((b_max*g_max,b_max))
    
    old_cont_ind = 0
    
    old_fork = models.fork.pairCounts
    models.fork.pairCounts = np.zeros((b_max*g_max,2))
    old_fork_dist = models.fork.dist
    models.fork.dist = np.zeros((b_max*g_max,2))
    
    for b in range(0, b_max):
        bg = b * g_max
        models.cont.pairCounts[bg:bg+g_max-1,:] = old_cont[old_cont_ind:old_cont_ind+g_max-1,:]
        models.cont.dist[bg:bg+g_max-1,:] = old_cont_dist[old_cont_ind:old_cont_ind+g_max-1,:]
        models.cont.dist[bg+g_max-1,0] = -np.inf
        models.cont.dist[bg+g_max-1,1:] = np.log10(sampler.sampleSimpleDirichlet(models.cont.alpha * models.cont.beta[1:]))
        
        models.fork.pairCounts[bg:bg+g_max-1,:] = old_fork[old_cont_ind:old_cont_ind+g_max-1,:]
        models.fork.dist[bg:bg+g_max-1,:] = old_fork_dist[old_cont_ind:old_cont_ind+g_max-1,:]
        models.fork.dist[bg+g_max-1,:] = np.log10(sampler.sampleSimpleBernoulli(sample.alpha_f * sample.beta_f))
        
        old_cont_ind = old_cont_ind + g_max - 1




def initialize_models(models, max_output, params, corpus_shape):
    global a_max, b_max, g_max

    ## One fork model:
    models.fork = Model(((g_max)*(b_max), 2))
    ## Two join models:
#    models.trans = Model((g_max*b_max, 2))
    models.reduce = Model((a_max, 2))
    
    ## One active model:
    models.act = Model((a_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape)
    models.root = Model((g_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape)
    models.root.beta = np.zeros(a_max)
    models.root.beta[1:] = np.ones(a_max-1) / (a_max-1)
    
    ## two awaited models:
    models.cont = Model(((g_max)*(b_max),b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
    models.start = Model(((a_max)*(a_max), b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
    models.cont.beta = np.zeros(b_max)
    models.cont.beta[1:] = np.ones(b_max-1) / (b_max-1)
    
    ## one pos model:
    models.pos = Model((b_max, g_max), alpha=float(params.get('alphag')), corpus_shape=corpus_shape)
    models.pos.beta = np.zeros(g_max)
    models.pos.beta[1:] = np.ones(g_max-1) / (g_max-1)
    
    ## one lex model:
    models.lex = Model((g_max, max_output+1))
    
    logging.debug("Value of amax=%d, b_max=%d, g_max=%d", a_max, b_max, g_max)
    return models

# Randomly initialize all the values for the hidden variables in the 
# sequence. Obeys constraints (e.g., when f=1,j=1 a=a_{t-1}) but otherwise
# samples randomly.
def initialize_state(ev_seqs, models):
    global a_max, b_max, g_max
    
    state_seqs = list()
    for sent_index,sent in enumerate(ev_seqs):
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
                    if random.random() >= 0.5:
                        state.f = 1
                    else:
                        state.f = 0
                    ## j is deterministic in the middle of the sentence
                    state.j = state.f
                    
                if state.f == 1 and state.j == 1:
                    state.a = prev_state.a
                else:
                    state.a = np.random.randint(1,a_max-1)

                state.b = np.random.randint(1,b_max-1)

            state.g = np.random.randint(1,g_max-1)
                    
            prev_state = state  
                            
            hid_seq.append(state)
            
        increment_counts(hid_seq, sent, models, sent_index)
        state_seqs.append(hid_seq)

    return state_seqs

def collect_trans_probs(hid_seqs, models):
    for sent_index,hid_seq in enumerate(hid_seqs):
        ## for every state transition in the sentence increment the count
        ## for the condition and for the output
        for index, state in enumerate(hid_seq):
            if index != 0:
                prevBG = bg_state(prevState.b, prevState.g)
                ## Count F & J
                if index == 1:
                    models.root.trans_prob[sent_index,index] = models.root.dist[prevState.g, state.a]

                    ## Count A & B
                    if state.f == 0 and state.j == 0:
                        models.root.trans_prob[sent_index,index] = models.act.trans_prob[sent_index,index] = models.act.dist[prevState.a, state.a]

                if state.j == 0:
                    aa = aa_state(prevState.a, state.a)
                    models.cont.trans_prob[sent_index,index] = models.start.trans_prob[sent_index,index] = models.start.dist[aa, state.b]
                else:
                    models.cont.trans_prob[sent_index,index] = models.cont.dist[prevBG, state.b]

            
            ## Count G
            models.pos.trans_prob[sent_index, index] = models.pos.dist[state.b, state.g]
            
            prevState = state

def increment_counts(hid_seq, sent, models, sent_index):
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
                aa = aa_state(prevState.a, state.a)
                models.start.count(aa, state.b)
            else:
                models.cont.count(prevBG, state.b)

            
        ## Count G
        models.pos.count(state.b, state.g)
            
        ## Count w
        models.lex.count(state.g, word)
        
        prevState = state
    
    prevBG = bg_state(hid_seq[-1].b, hid_seq[-1].g)
## WS: REMOVED THESE: WAS DISTORTING OUTPUTS BC F MODEL NOT REALLY CONSULTED AT END (MODEL ACTUALLY KNOWS ITS AT END)
#    models.fork.count(prevBG, 0)
#    models.reduce.count(hid_seq[-1].a, 1)

def bg_state(b, g):
    global g_max
    return b*(g_max) + g

def aa_state(prevA, a):
    global a_max
    return prevA * (a_max) + a
    
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
    
    ## Make sure all returned values are ints:
    return map(int, (f_ind,j_ind,a_ind, b_ind, g_ind))

def getStateIndex(f,j,a,b,g):
    global a_max, b_max, g_max
    return (((f*2 + j)*a_max + a) * b_max + b)*g_max + g

def getStateRange(f,j,a,b):
    global g_max
    start = getStateIndex(f,j,a,b,0)
    return range(start,start+g_max)

def getGmax():
    global g_max
    return g_max
    
def getAmax():
    global a_max
    return a_max
    
def getBmax():
    global b_max
    return b_max