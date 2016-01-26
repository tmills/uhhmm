#!/usr/bin/env python3.4

import logging
import time
import numpy as np
import ihmm_sampler as sampler
import ihmm_io
import pdb
import socket
import sys
import tempfile
from multiprocessing import Process,Queue,JoinableQueue
import zmq
from PyzmqSentenceDistributerServer import *
from PyzmqMessage import *
from PyzmqWorker import *

# The set of random variable values at one word
# There will be one of these for every word in the training set
class State:
    def __init__(self, depth, state=None):
        self.depth = depth
        self.f = [None] * depth
        self.j = [None] * depth
        self.a = [None] * depth
        self.b = [None] * depth
        
        if state == None:
            for d in range(0,depth):
                self.f[d] = -1
                self.j[d] = -1
                self.a[d] = 0
                self.b[d] = 0
            self.g = 0
        else:
            (self.f, self.j, self.a, self.b, self.g) = state.f, state.j, state.a, state.b, state.g

    def str(self):
        string = ''
        for d in range(0, self.depth):
            ## only one depth position of f and j will be active:
            if self.f[d] >= 0 and self.j[d] >= 0:
                f_str = '+/' if self.f[d]==1 else '-/'        
                string += f_str
                j_str = '+' if self.j[d]==1 else '-'
                string += j_str
            else:
                string += "-1/-1"

        string += "::"
        
        for d in range(0, self.depth):
            if d > 0:
                string += ";"

            if self.a[d] > 0:
                string += "ACT" + str(self.a[d]) + '/AWA' + str(self.b[d])
        
            
        string += ':POS' + str(self.g)
        
        return string

    def to_list(self):
        return (self.f, self.j, self.a, self.b, self.g)

    def max_fork_depth(self):
        for d in range(0, self.depth):
            if self.f[d] >= 0:
                return d
                
    def max_awa_depth(self):
        for d in range(0, self.depth):
            if self.b[d] == 0:
                return d-1
                
        return 0
        
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
        out_counts = self.pairCounts[...,out]
        out_counts[cond] += 1

    def dec(self, cond, out):
        self.pairCounts[cond,out] -= 1

    def selfSampleDirichlet(self):
        self.sampleDirichlet(self.alpha * self.beta)
        
    def sampleDirichlet(self, base):
        self.dist = sampler.sampleDirichlet(self.pairCounts, base)
        self.pairCounts[:] = 0

    def sampleBernoulli(self, base):
        self.dist = sampler.sampleBernoulli(self.pairCounts, base)
        self.pairCounts[:] = 0
        
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
def sample_beam(ev_seqs, params, report_function, checkpoint_function, working_dir, pickle_file=None):    
    
    global start_a, start_b, start_g
    global a_max, b_max, g_max
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))

    
    num_samples = 0
    depth = int(params.get('depth', 1))
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    max_samples = int(params.get('num_samples'))
    num_procs = int(params.get('num_procs', 0))
    debug = params.get('debug', 'INFO')
    profile = bool(int(params.get('profile', 0)))
    finite = bool(int(params.get('finite', 0)))
    cluster_cmd = params.get('cluster_cmd', None)
    infinite_sample_prob = float(params.get('infinite_prob', 0.0))
    return_to_finite = False
    
    logging.basicConfig(level=getattr(logging, debug))
    logging.info("Starting beam sampling")

    seed = int(params.get('seed', -1))
    if seed > 0:
        logging.info("Using seed %d for random number generator." % (seed))
        np.random.seed(seed)
    else:
        logging.info("Using default seed for random number generator.")
 
    if not profile:
        logging.info('profile is set to %s, importing and installing pyx' % profile)    
        import pyximport; pyximport.install()

    import DepthOneInfiniteSampler
    import HmmSampler
    import FullDepthCompiler

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

        models = initialize_models(models, max_output, params, (len(ev_seqs), maxLen), depth, a_max, b_max, g_max)
        hid_seqs = initialize_state(ev_seqs, models, depth)

        sample = Sample()
    #    sample.hid_seqs = hid_seqs
        sample.alpha_a = models.root[0].alpha ## float(params.get('alphaa'))
        sample.alpha_b = float(params.get('alphab'))
        sample.alpha_g = float(params.get('alphag'))
        sample.alpha_f = float(params.get('alphaf'))
        sample.alpha_j = float(params.get('alphaj'))
        ## use plus 2 here (until moved later) since we need the null state (0) as well as 
        ## the extra part of the stick for "new tables"
        sample.beta_a = models.root[0].beta ## np.ones((1,start_a+2)) / start_a
    #    sample.beta_a[0][0] = 0
        sample.beta_b = models.cont[0].beta
    #    sample.beta_b[0] = 0
        sample.beta_g = models.pos.beta
        sample.beta_f = np.ones(2) / 2
        sample.beta_j = np.ones(2) / 2
        sample.gamma = float(params.get('gamma'))
        sample.discount = float(params.get('discount'))
        sample.ev_seqs = ev_seqs
    
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.sampleDirichlet(params['h'])
        models.pos.selfSampleDirichlet()
        if np.argwhere(np.isnan(models.pos.dist)).size > 0:
            logging.error("Resampling the pos dist resulted in a nan")
        
        for d in range(0, depth):
            models.start[d].sampleDirichlet(sample.alpha_b * sample.beta_b)
            models.cont[d].sampleDirichlet(sample.alpha_b * sample.beta_b)
            models.act[d].sampleDirichlet(sample.alpha_a * sample.beta_a)
            models.root[d].sampleDirichlet(sample.alpha_a * sample.beta_a)
            models.reduce[d].sampleBernoulli(sample.alpha_j * sample.beta_j)
            models.fork[d].sampleBernoulli(sample.alpha_f * sample.beta_f)
    
        sample.models = models
        iter = 0

    else:
        sample = ihmm_io.read_serialized_sample(pickle_file)
        sample.log_prob = 0
        models = sample.models
        hid_seqs = sample.hid_seqs
        sample.hid_seqs = [] ## Empty out hid_seqs because we will append later.
        
        a_max = models.act[0].dist.shape[-1]
        b_max = models.cont[0].dist.shape[-1]
        g_max = models.pos[0].dist.shape[-1]
        
        iter = sample.iter+1

    collect_trans_probs(hid_seqs, models)
    
    stats = Stats()
    inf_procs = dict()
    
    logging.debug(ev_seqs[0])
    logging.debug(list(map(lambda x: x.str(), hid_seqs[0])))

    workDistributer = PyzmqSentenceDistributerServer(ev_seqs, working_dir)
    
    logging.info("Start a new worker with python3 scripts/PyzmqWorker.py %s %d %d %d %d" % (workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port, maxLen+1))
    
    ## Initialize all the sub-processes with their input-output queues
    ## and dimensions of matrix they'll need    
    if num_procs > 0:
        for cur_proc in range(0,num_procs):
            ## Initialize and start the sub-process
            inf_procs[cur_proc] = PyzmqWorker(workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port, maxLen+1, out_freq=100, tid=cur_proc)
            inf_procs[cur_proc].start()
    
    elif cluster_cmd != None:
        start_workers(workDistributer, cluster_cmd, maxLen)
            
    logging.info("Starting workers")

    ### Start doing actual sampling:
    while num_samples < max_samples:
        sample.iter = iter

        if finite:
            if np.random.random() < infinite_sample_prob:
                if finite:
                    logging.info("Randomly chose infinite sample with probability %f" % infinite_sample_prob)
                finite = False
                return_to_finite = True
            else:
                logging.info("Performing standard finite sample")

        if iter > 0 and not finite:
        
            ## now that we've resampled models, store the transition probabilities that
            ## the new model assigned to all the transitions
            models.pos.u =  models.pos.trans_prob +  np.log10(np.random.random((len(ev_seqs), maxLen)) )
            models.root.u = models.root.trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )
            models.cont.u = models.cont.trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )

            ## Break off the beta sticks before actual processing -- instead of determining during
            ## inference whether to create a new state we use an auxiliary variable u to do it
            ## ahead of time, but note there is no guarantee we will ever use it.
            ## TODO: Resample beta, which will allow for unused probability mass to go to the end again?
            while a_max < 20 and models.root.u.min() < max(models.root.dist[...,-1].max(),models.act.dist[...,-1].max()):
                logging.info('Breaking a stick')
                break_a_stick(models, sample, params)
                a_max = models.act.dist.shape[-1]
                
            if a_max >= 20:
                logging.warning('Stick-breaking (a) terminated due to hard limit and not gracefully.')
        
            while b_max < 20 and models.cont.u.min() < max(models.cont.dist[...,-1].max(), models.start.dist[...,-1].max()):
                logging.info('Breaking b stick')
                break_b_stick(models, sample, params)
                b_max = models.cont.dist.shape[-1]

            if b_max >= 50:
                logging.warning('Stick-breaking (b) terminated due to hard limit and not gracefully.')

            while g_max < 50 and models.pos.u.min() < models.pos.dist[...,-1].max():
                logging.info('Breaking g stick')
                break_g_stick(models, sample, params)
                g_max = models.pos.dist.shape[-1]
                if np.argwhere(np.isnan(models.pos.dist)).size > 0:
                    logging.error("Breaking the g stick resulted in a nan in the output distribution")
            
            if g_max >= 50:
                logging.warning('Stick-breaking (g) terminated due to hard limit and not gracefully.')


        ## These values keep track of actual maxes not user-specified --
        ## so if user specifies 10 to start this will be 11 because of state 0 (init)
        a_max = models.act[0].dist.shape[-1]
        b_max = models.cont[0].dist.shape[-1]
        g_max = models.pos.dist.shape[-1]
        
        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = 2 * 2 * a_max * b_max * g_max

        logging.info("Number of a states=%d, b states=%d, g states=%d, total=%d" % (a_max-2, b_max-2, g_max-2, totalK))
        
        ## Give the models to the model server. (Note: We used to pass them to workers via temp files which works when you create new
        ## workers but is harder when you need to coordinate timing of reading new models in every pass)
        t0 = time.time()
        if finite:
            (trans_mat, obs_mat) = FullDepthCompiler.FullDepthCompiler(1).compile_models(models)
            workDistributer.run_one_iteration((models, trans_mat, obs_mat), finite)
        else:
            workDistributer.run_one_iteration(models, finite)
        
        ## Wait for server to finish distributing sentences for this iteration:
        t1 = time.time()
        
        logging.info("Sampling time for iteration %d is %d s" % (iter, t1-t0))

        ## Read the output and put it in a map -- use a map because sentences will
        ## finish asynchronously -- keep the sentence index so we can extract counts
        ## in whatever order we want (without aligning to sentences) but then order
        ## the samples so that when we print them out they align for inspection.
        t0 = time.time()
        num_processed = 0
        parses = workDistributer.get_parses()
        sample_map = dict()
        
        for parse in parses:
            num_processed += 1
            if parse.success:
                increment_counts(parse.state_list, ev_seqs[ parse.index ], models, parse.index)
                sample.log_prob += parse.log_prob
            
            sample_map[parse.index] = parse.state_list

        if num_processed < len(ev_seqs):
            logging.warning("Didn't receive the correct number of parses at iteration %d" % iter)
        
        ## samples got unsorted by queueing them so resort them just for the purpose 
        ## of debugging.
        for key in sorted(sample_map.keys()):
            sample.hid_seqs.append(sample_map[key])

        t1 = time.time()
        logging.info("Building counts tables took %d s" % (t1-t0))
        
        checkpoint_function(sample) 
        if iter >= burnin and (iter-burnin) % iters == 0:
            #samples.append(sample)
            report_function(sample)
            logging.info(".\n")
            num_samples += 1
        
        t0 = time.time()

        next_sample = Sample()
                
        ## Sample hyper-parameters
        ## This is, e.g., where we might add categories to the a,b,g variables with
        ## stick-breaking. Without that, the values will stay what they were 
        next_sample.alpha_f = sample.alpha_f
        next_sample.beta_f = sample.beta_f
        next_sample.alpha_j = sample.alpha_j
        next_sample.beta_j = sample.beta_j
        
        next_sample.alpha_a = models.root[0].alpha
        next_sample.beta_a = models.root[0].beta
        next_sample.alpha_b = models.cont[0].alpha
        next_sample.beta_b = models.cont[0].beta
#        next_sample.alpha_g = sample.alpha_g
#        next_sample.beta_g = sample.beta_g
        next_sample.alpha_g = models.pos.alpha
        next_sample.beta_g = models.pos.beta
        next_sample.gamma = sample.gamma
        next_sample.ev_seqs = ev_seqs
        
        prev_sample = sample
        sample = next_sample


        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        ## After stick-breaking we probably need to re-sample all the models:
        remove_unused_variables(models)
        resample_beta_g(models, sample.gamma)
               
        models.lex.sampleDirichlet(params['h'])
        models.pos.selfSampleDirichlet()

        for d in range(0, depth):        
            models.start[d].sampleDirichlet(sample.alpha_b * sample.beta_b)
            models.cont[d].sampleDirichlet(sample.alpha_b * sample.beta_b)
            models.act[d].sampleDirichlet(sample.alpha_a * sample.beta_a)
            models.root[d].sampleDirichlet(sample.alpha_a * sample.beta_a)
            models.reduce[d].sampleBernoulli(sample.alpha_j * sample.beta_j)
            models.fork[d].sampleBernoulli(sample.alpha_f * sample.beta_f)
        
        collect_trans_probs(hid_seqs, models)

        t1 = time.time()
        
        logging.debug("Resampling models took %d s" % (t1-t0))       
        
        sample.models = models
        
        if not finite and return_to_finite:
            finite = True
            return_to_finite = False
        
        iter += 1
        
    logging.debug("Ending sampling")
    workDistributer.stop()

    for cur_proc in range(0,num_procs):
        inf_procs[cur_proc].join()
        inf_procs[cur_proc] = None
        
    return (samples, stats)

def remove_unused_variables(models):
    ## Have to check if it's greater than 2 -- empty state (0) and final state (-1) are not supposed to have any counts:
    while sum(models.pos.pairCounts.sum(0)==0) > 2:
        remove_pos_from_models(models, np.where(models.pos.pairCounts.sum(0)==0)[0][1])

def remove_pos_from_models(models, pos):
    ## delete the 1st dimension from the fork distribution:
    models.fork.pairCounts = np.delete(models.fork.pairCounts, pos, 1)
    models.fork.dist = np.delete(models.fork.dist, pos, 1)
    
    models.root.pairCounts = np.delete(models.root.pairCounts, pos, 0)
    models.root.dist = np.delete(models.root.dist, pos, 0)
    
    models.cont.pairCounts = np.delete(models.cont.pairCounts, pos, 1)
    models.cont.dist = np.delete(models.cont.dist, pos, 1)
    
    ## for the given POS index, delete the 1st (output) dimension from the
    ## counts and distribution of p(pos | awa)
    models.pos.pairCounts = np.delete(models.pos.pairCounts, pos, 1)
    models.pos.dist = np.delete(models.pos.dist, pos, 1)
    
    ## for the given POS index, delete the 0th (input) dimension from the
    ## counts and distribution of p(lex | pos)
    models.lex.pairCounts = np.delete(models.lex.pairCounts, pos, 0)
    models.lex.dist = np.delete(models.lex.dist, pos, 0)

    ## Shorten the beta stick as well:
    models.pos.beta = np.delete(models.pos.beta, pos, 0)

def add_model_column(model):

    model.pairCounts = np.insert(model.pairCounts, model.pairCounts.shape[-1], np.zeros(model.pairCounts.shape[0:-1]), model.pairCounts.ndim-1)
    
    param_a = np.tile(model.alpha * model.beta[-2], model.dist.shape[0:-1])
    param_b = model.alpha * (1 - model.beta[0:-1].sum())

    if param_b < 0:
        param_a_str = str(param_a)
        param_b_str = str(param_b)
        logging.error("param b is < 0! inputs are alpha=%f, beta=%s, betasum=%f" % (model.alpha, str(model.beta), model.beta[0:-1].sum()))
        logging.info("Param a=%s and param b=%s" % (param_a_str, param_b_str))
        param_b = 0

    if param_a.min() < 1e-2 or param_b < 1e-2:
        pg = np.random.binomial(1, param_a / (param_a+param_b))
    else:
        pg = np.random.beta(param_a, param_b)
    
    ## Copy the last dimension into the new dimension:
    model.dist = np.insert(model.dist, model.dist.shape[-1], model.dist[...,-1], model.dist.ndim-1)
    model.dist[...,-2] += np.log10(pg)
    model.dist[...,-1] += np.log10(1-pg)
    
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
    model.beta = np.append(model.beta, np.zeros(1))
    old_portion = new_portion = 0
    
    while old_portion == 0 or new_portion == 0:
        old_group_fraction = np.random.beta(1, gamma)
        old_portion = old_group_fraction * beta_end
        new_portion = (1-old_group_fraction) * beta_end
    
    model.beta[-2] = old_portion
    model.beta[-1] = new_portion

def break_a_stick(models, sample, params):

    a_max = models.root.dist.shape[-1]
    b_max = models.cont.dist.shape[-1]
    
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
    
    ## Add a row to both dimensions of the model input
    ## need to modify shape to do appends so we'll make it a list from a tuple:
    ## Resize the distributions in the same way but we'll also need to initialize 
    ## them to non-zero values
    models.start.pairCounts = np.append(models.start.pairCounts, np.zeros((1,a_max,b_max)), 0)
    models.start.dist = np.append(models.start.dist, np.zeros((1,a_max,b_max)), 0)
    models.start.dist[a_max,...] = sampler.sampleDirichlet(models.start.pairCounts[a_max,...], sample.alpha_b * sample.beta_b)

    models.start.pairCounts = np.append(models.start.pairCounts, np.zeros((a_max+1,1,b_max)), 1)  
    models.start.dist = np.append(models.start.dist, np.zeros((a_max+1,1,b_max)), 1) 
    models.start.dist[:,a_max,:] = sampler.sampleDirichlet(models.start.pairCounts[:,a_max,:], sample.alpha_b * sample.beta_b)

def break_b_stick(models, sample, params):
    
    b_max = models.cont.dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    
    ## Keep the stick with cont and copy it over to beta
    break_beta_stick(models.cont, sample.gamma)
    models.start.beta = models.cont.beta
    
    ## Add a column to both output distributions:
    add_model_column(models.cont)
    add_model_column(models.start)
    
    ## Add a row to the POS output distribution which depends only on b:
    add_model_row_simple(models.pos, models.pos.alpha * models.pos.beta[1:])

    models.fork.pairCounts = np.append(models.fork.pairCounts, np.zeros((1,g_max,2)), 0)
    models.fork.dist = np.append(models.fork.dist, np.zeros((1,g_max,2)), 0)
    models.fork.dist[b_max,...] = sampler.sampleBernoulli(models.fork.pairCounts[b_max,...], sample.alpha_f * sample.beta_f)
    
    models.cont.pairCounts = np.append(models.cont.pairCounts, np.zeros((1,g_max,b_max+1)), 0)
    models.cont.dist = np.append(models.cont.dist, np.zeros((1,g_max,b_max+1)),0)
    models.cont.dist[b_max,...] = sampler.sampleDirichlet(models.cont.pairCounts[b_max,...], models.cont.alpha * models.cont.beta)
    
def break_g_stick(models, sample, params):

    b_max = models.cont.dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    
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
    
    models.fork.pairCounts = np.append(models.fork.pairCounts, np.zeros((b_max,1,2)), 1)
    models.fork.dist = np.append(models.fork.dist, np.zeros((b_max,1,2)), 1)
    models.fork.dist[:,g_max,:] = sampler.sampleBernoulli(models.fork.pairCounts[:,g_max,:], sample.alpha_f * sample.beta_f)
    
    models.cont.pairCounts = np.append(models.cont.pairCounts, np.zeros((b_max,1,b_max)), 1)
    models.cont.dist = np.append(models.cont.dist, np.zeros((b_max,1,b_max)), 1)
    models.cont.dist[:,g_max,:] = sampler.sampleDirichlet(models.cont.pairCounts[:,g_max,:], models.cont.alpha * models.cont.beta)

def resample_beta_g(models, gamma):
    
    b_max = models.cont[0].dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    m = np.zeros((b_max,g_max-1))
    
    for b in range(0, b_max):
        for g in range(0, g_max-1):
            if models.pos.pairCounts[b][g] == 0:
                m[b][g] = 0
            
            ## (rand() < (ialpha0 * ibeta(k)) / (ialpha0 * ibeta(k) + l - 1));
            else:
                for l in range(1, int(models.pos.pairCounts[b][g])+1):
                    dart = np.random.random()
                    alpha_beta = models.pos.alpha * models.pos.beta[g]
                    m[b][g] += (dart < (alpha_beta / (alpha_beta + l - 1)))

    if 0 in m.sum(0)[1:]:
        logging.warning("There seems to be an illegal value here:")
        
    params = np.append(m.sum(0)[1:], gamma)
    models.pos.beta[1:] = 0
    models.pos.beta[1:] += sampler.sampleSimpleDirichlet(params)

def initialize_models(models, max_output, params, corpus_shape, depth, a_max, b_max, g_max):

    models.fork = [None] * depth
    ## J models:
    models.trans = [None] * depth
    models.reduce = [None] * depth
    ## Active models:
    models.act = [None] * depth
    models.root = [None] * depth
    ## Reduce models:
    models.cont = [None] * depth
    models.exp = [None] * depth
    models.next = [None] * depth
    models.start = [None] * depth
    
    ## One fork model:    
    for d in range(0, depth):
        models.fork[d] = Model((b_max, g_max, 2))
    
        ## Two join models:
        models.trans[d] = Model((b_max, g_max, 2))
        models.reduce[d] = Model((a_max, 2))
    
        ## TODO -- set d > 0 beta to the value of the model at d (can probably do this later)
        ## One active model:
        models.act[d] = Model((a_max, b_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape)
        models.root[d] = Model((b_max, g_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape)
        models.root[d].beta = np.zeros(a_max)
        models.root[d].beta[1:] = np.ones(a_max-1) / (a_max-1)
    
        ## four awaited models:
        models.cont[d] = Model((b_max, g_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
        models.exp[d] = Model((g_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
        models.next[d] = Model((b_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
        models.start[d] = Model((a_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape)
        models.cont[d].beta = np.zeros(b_max)
        models.cont[d].beta[1:] = np.ones(b_max-1) / (b_max-1)
        
    
    ## one pos model:
    models.pos = Model((b_max, g_max), alpha=float(params.get('alphag')), corpus_shape=corpus_shape)
    models.pos.beta = np.zeros(g_max)
    models.pos.beta[1:] = np.ones(g_max-1) / (g_max-1)
    
    ## one lex model:
    models.lex = Model((g_max, max_output+1))
    
    return models

# Randomly initialize all the values for the hidden variables in the 
# sequence. Obeys constraints (e.g., when f=1,j=1 a=a_{t-1}) but otherwise
# samples randomly.
def initialize_state(ev_seqs, models, depth):
    global a_max, b_max, g_max
    
    state_seqs = list()
    for sent_index,sent in enumerate(ev_seqs):
        hid_seq = list()
        for index,word in enumerate(sent):
            state = State(depth)
            ## special case for first word
            if index == 0:
                state.f[0] = 0
                state.j[0] = 0
                state.a[0] = 0
                state.b[0] = 0
            else:
                if index == 1:
                    state.f[0] = 1
                    state.j[0] = 0
                else:
                    if np.random.random() >= 0.5:
                        state.f[0] = 1
                    else:
                        state.f[0] = 0
                    ## j is deterministic in the middle of the sentence
                    state.j[0] = state.f[0]
                    
                if state.f == 1 and state.j == 1:
                    state.a[0] = prev_state.a[0]
                else:
                    state.a[0] = np.random.randint(1,a_max-1)

                state.b[0] = np.random.randint(1,b_max-1)

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
            d = state.max_fork_depth()
            
            if index != 0:
                if index == 1:
                    models.root[0].trans_prob[sent_index, index] = models.root[0].dist[0, prevState.g, state.a[0]]
                    models.start[0].trans_prob[sent_index, index] = models.start[0].dist[0, state.a[0], state.b[0]]
                else:
                    ## Fork and join don't have trans_probs because they are finite:
                    if state.f[d] == 0 and state.j[d] == 0:
                        models.act[d].trans_prob[sent_index, index] = models.root[d].trans_prob[sent_index, index] = models.act[d].dist[ prevState.a[d], prevState.b[d], state.a[d] ]
                        models.start[d].trans_prob[sent_index, index] = models.cont[d].trans_prob[sent_index, index] = models.start[d].dist[ prevState.a[d], state.a[d], state.b[d] ]
                    elif state.f[d] == 1 and state.j[d] == 1:
                        models.act[d].trans_prob[sent_index, index] = models.root[d].trans_prob[sent_index, index] = 0
                        models.cont[d].trans_prob[sent_index, index] = models.start[d].trans_prob[sent_index, index] = models.cont[d].dist[ prevState.b[d], prevState.g, state.b[d] ]
                    elif state.f[d] == 1 and state.j[d] == 0:
                        models.root[d+1].trans_prob[sent_index, index] = models.act[d].trans_prob[sent_index, index]  = models.root[d+1].dist[ prevState.a[d], prevState.g, state.a[d+1] ]
                    elif state.f[d] == 0 and state.j[d] == 1:
                        models.act[d-1].trans_prob[sent_index, index] = models.root[d-1].trans_prob[sent_index, index] = 0
                        models.cont[d].trans_prob[sent_index, index] = models.start[d].trans_prob[sent_index, index] = models.cont[d-1].dist[ prevState.b[d-1], prevState.g, state.b[d] ]
                                                    
            prevState = state

def increment_counts(hid_seq, sent, models, sent_index):
    depth = len(models.fork)
    
    ## for every state transition in the sentence increment the count
    ## for the condition and for the output
    for index,word in enumerate(sent):
        state = hid_seq[index]
        d = state.max_fork_depth()
        
        if index != 0:
            ## Count F & J
            if index == 1:
                ## No counts for f & j -- deterministically +/- at depth 0
                models.root[0].count((0, prevState.g), state.a[0])
                models.start[0].count((0, state.a[0]), state.b[0])
            else:
                models.fork[d].count((prevState.b[d], prevState.g), state.f[d])

                if state.f[d] == 0:
                    models.reduce[d].count(prevState.a[d], state.j[d])
                elif state.f[d] == 1:
                    models.trans[d].count((prevState.b[d], prevState.g), state.j[d])
                            
                ## Count A & B
                if state.f[d] == 0 and state.j[d] == 0:
                    prevB = prevState.b[d-1] if d > 0 else 0
                    models.act[d].count((prevState.a[d], prevB), state.a[d])
                    models.start[d].count((prevState.a[d], state.a[d]), state.b[d])
                elif state.f[d] == 1 and state.j[d] == 1:
                    ## no change to act, awa increments cont model
                    models.cont[d].count((prevState.b[d], prevState.g), state.b[d])
                elif state.f[d] == 1 and state.j[d] == 0:
                    ## run root and exp models at depth d+1
                    models.root[d+1].count((prevState.b[d-1], prevState.g), state.a[d+1])
                    models.exp[d+1]((prevState.g, state.a[d+1]), state.b[d+1])
                elif state.f[d] == 0 and state.j[d] == 1:
                    ## lower level finished -- awaited can transition
                    models.next[d-1].count((prevState.b[d-1], state.a[d]), state.b[d-1])
                         
        
            ## Count G
            awa_depth = state.max_awa_depth()       
            models.pos.count(state.b[awa_depth], state.g)
        else:
            models.pos.count(0, state.g)
                    
        ## Count w
        models.lex.count(state.g, word)
        
        prevState = state
    
#    prevBG = bg_state(hid_seq[-1].b, hid_seq[-1].g)
## WS: REMOVED THESE: WAS DISTORTING OUTPUTS BC F MODEL NOT REALLY CONSULTED AT END (MODEL ACTUALLY KNOWS ITS AT END)
#    models.fork.count(prevBG, 0)
#    models.reduce.count(hid_seq[-1].a, 1)

def getGmax():
    global g_max
    return g_max
    
def getAmax():
    global a_max
    return a_max
    
def getBmax():
    global b_max
    return b_max
