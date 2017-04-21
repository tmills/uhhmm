#!/usr/bin/env python3.4

import logging
import time
import numpy as np
import distribution_sampler as sampler
import uhhmm_io
import pdb
import signal
import socket
import sys
import tempfile
from multiprocessing import Process,Queue,JoinableQueue
import zmq
from WorkDistributerServer import WorkDistributerServer
from PyzmqMessage import *
#from PyzmqWorker import PyzmqWorker
from State import sentence_string
import State
from Indexer import Indexer
import multiprocessing as mp
import FullDepthCompiler
import NoopCompiler
import DepthOneInfiniteSampler
import DistributedModelCompiler
import HmmSampler
from dahl_split_merge import perform_split_merge_operation
from models import Model, Models
from workers import start_local_workers_with_distributer, start_cluster_workers
from pcfg_translator import pcfg_increment_counts, calc_anneal_alphas

# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0

# Histogram of how many instances of each state you have random sampled
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

# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
def sample_beam(ev_seqs, params, report_function, checkpoint_function, working_dir, pickle_file=None, gold_seqs=None, input_seqs_file=None):

    global start_a, start_b, start_g
    global a_max, b_max, g_max
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))
    assert start_a == start_b and start_b == start_g, "A, B and G must have the same domain size!"
    sent_lens = list(map(len, ev_seqs))
    total_sent_lens = sum(sent_lens)
    maxLen = max(map(len, ev_seqs))
    max_output = max(map(max, ev_seqs))
    num_sents = len(ev_seqs)
    num_tokens = np.sum(sent_lens)

    num_samples = 0
    ## Set debug first so we can use it during config setting:
    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile','')
    if logfile:
        logging.basicConfig(level=getattr(logging, debug),filename=logfile)
    else:
        logging.basicConfig(level=getattr(logging, debug), stream=sys.stdout)

    depth = int(params.get('depth', 1))
    init_depth = int(params.get('init_depth', depth))
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    max_samples = int(params.get('num_samples'))
    num_cpu_workers = int(params.get('cpu_workers', 0))
    num_procs = int(params.get('num_procs', -1))
    if num_procs != -1:
        num_cpu_workers = num_procs
        logging.warn("num_procs config is deprecated for cpu_workers and gpu_workers configs. Treating num_procs as num_cpu_workers=%d" % (num_cpu_workers))

    num_gpu_workers = int(params.get('gpu_workers', 0))

    profile = bool(int(params.get('profile', 0)))
    finite = bool(int(params.get('finite', 0)))
    cluster_cmd = params.get('cluster_cmd', None)
    split_merge_iters = int(params.get('split_merge_iters', -1))
    infinite_sample_prob = float(params.get('infinite_prob', 0.0))
    batch_size = min(num_sents, int(params.get('batch_size', num_sents)))
    gpu = bool(int(params.get('gpu', 0)))
    gpu_batch_size = min(num_sents, int(params.get('gpu_batch_size', 32 if gpu == 1 else 1)))
    if gpu and num_gpu_workers < 1 and num_cpu_workers > 0:
        logging.warn("Inconsistent config: gpu flag set with %d gpu workers; setting gpu=False" % (num_gpu_workers))
        gpu=False
    init_tempature = float(params.get('init_temperature', 0))

    return_to_finite = False
    ready_for_sample = False

    logging.info("Starting beam sampling")

    if (gold_seqs != None and 'num_gold_sents' in params):
        logging.info('Using gold tags for %s sentences.' % str(params['num_gold_sents']))

    seed = int(params.get('seed', -1))
    if seed > 0:
        logging.info("Using seed %d for random number generator." % (seed))
        np.random.seed(seed)
    else:
        logging.info("Using default seed for random number generator.")

    samples = []
    models = Models()
    start_ind = 0
    end_ind = min(num_sents, batch_size)

    logging.info("Initializing state")

    if pickle_file == None:
        ## Add 1 to every start value for "Null/start" state
        a_max = start_a+2
        b_max = start_b+2
        g_max = start_g+2

        models = initialize_models(models, max_output, params, (len(ev_seqs), maxLen), depth, a_max, b_max, g_max)
        if input_seqs_file is None:
            hid_seqs = [None] * len(ev_seqs)
        else:
            logging.info("Trainer was initialized with input sequences from a previous run.")
            hid_seqs = initialize_and_load_state(ev_seqs, models, depth, uhhmm_io.read_input_states(input_seqs_file, depth))
            max_state_check(hid_seqs, models, "initialization")

        sample = Sample()
        sample.alpha_f = models.F[0].alpha
        sample.alpha_j = models.J[0].alpha
        sample.alpha_a = models.A[0].alpha
        sample.alpha_b = models.B_J0[0].alpha
        sample.alpha_g = models.pos.alpha
        sample.alpha_h = models.lex.alpha
        
        sample.beta_f = models.F[0].beta
        sample.beta_j = models.J[0].beta
        sample.beta_a = models.A[0].beta
        sample.beta_b = models.B_J0[0].beta
        sample.beta_g = models.pos.beta
        sample.beta_h = models.lex.beta
        sample.gamma = float(params.get('gamma'))
        sample.discount = float(params.get('discount'))
        sample.ev_seqs = ev_seqs

        RB_INIT = False
        if RB_INIT:
            pcfg_increment_counts(None, None, models, RB_init=RB_INIT)

        resample_all(models, sample, params, depth)
        models.resetAll()

        sample.models = models
        iter = 0

    else:
        sample = uhhmm_io.read_serialized_sample(pickle_file)
        sample.log_prob = 0
        models = sample.models
        hid_seqs = sample.hid_seqs
        max_state_check(hid_seqs, models, "reading sample from pickle file")

        pos_counts = models.pos.pairCounts[:].sum()
        lex_counts = models.lex.pairCounts[:].sum()

        a_max = models.A[0].dist.shape[-1]
        b_max = models.B_J0[0].dist.shape[-1]
        g_max = models.pos.dist.shape[-1]

        models.resetAll()

        iter = sample.iter+1

    # import pickle
    # fixed_model = working_dir+'/fixed.models.bin'
    # if os.path.exists(fixed_model):
    #     models = pickle.load(open(fixed_model, 'rb'))
    # else:
    #     pickle.dump(models, open(fixed_model, 'wb'))

    indexer = Indexer(models)
    if not finite:
        collect_trans_probs(hid_seqs, models, 0, num_sents)

    stats = Stats()
    inf_procs = list()

    workDistributer = WorkDistributerServer(ev_seqs, working_dir)
    logging.info("GPU is %s with %d workers and batch size %d" % (gpu, num_gpu_workers, gpu_batch_size) )
    logging.info("Start a new worker with python3 scripts/workers.py %s %d %d %d %d %d %d" % (workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port, maxLen+1, int(gpu), gpu_batch_size))

    ## Initialize all the sub-processes with their input-output queues
    ## and dimensions of matrix they'll need
    if num_cpu_workers+num_gpu_workers > 0:
        inf_procs = start_local_workers_with_distributer(workDistributer, maxLen, num_cpu_workers, num_gpu_workers, gpu, gpu_batch_size)
        signal.signal(signal.SIGINT, lambda x,y: handle_sigint(x,y, inf_procs))

    elif cluster_cmd != None:
        start_cluster_workers(workDistributer, cluster_cmd, maxLen, gpu)
    else:
        master_config_file = './masterConfig.txt'
        with open(master_config_file, 'w') as c:
            print(' '.join([str(x) for x in [workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port, maxLen+1, int(gpu), gpu_batch_size]]), file=c)
            print('OK', file=c)

    logging.info("Starting workers")

    ### Start doing actual sampling:
    while num_samples < max_samples:
        sample.iter = iter

        ## If user specified an init depth (i.e. less than the final depth), increment it after 1000 iterations.
        if init_depth < depth:
            if iter > 0 and iter % 1000 == 0:
              init_depth += 1

        split_merge = False
        if split_merge_iters > 0 and iter >= burnin and (iter-burnin) % split_merge_iters == 0:
            split_merge = True

        if finite:
            if iter > 0 and np.random.random() < infinite_sample_prob:
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
            models.root[0].u = models.root[0].trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )
            models.cont[0].u = models.cont[0].trans_prob + np.log10(np.random.random((len(ev_seqs), maxLen)) )

            ## Break off the beta sticks before actual processing -- instead of determining during
            ## inference whether to create a new state we use an auxiliary variable u to do it
            ## ahead of time, but note there is no guarantee we will ever use it.
            ## TODO: Resample beta, which will allow for unused probability mass to go to the end again?
            while a_max < 20 and models.root[0].u.min() < max(models.root[0].dist[...,-1].max(),models.act[0].dist[...,-1].max()):
                logging.info('Breaking a stick')
                break_a_stick(models, sample, params)
                a_max = models.act[0].dist.shape[-1]

            if a_max >= 20:
                logging.warning('Stick-breaking (a) terminated due to hard limit and not gracefully.')

            while b_max < 20 and models.cont[0].u.min() < max(models.cont[0].dist[...,-1].max(), models.start[0].dist[...,-1].max()):
                logging.info('Breaking b stick')
                break_b_stick(models, sample, params)
                b_max = models.cont[0].dist.shape[-1]

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
        a_max = models.A[0].dist.shape[-1]
        b_max = models.B_J0[0].dist.shape[-1]
        g_max = models.pos.dist.shape[-1]

        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = indexer.get_state_size()

        logging.info("Number of a states=%d, b states=%d, g states=%d, total=%d, start_ind=%d, end_ind=%d" % (a_max-2, b_max-2, g_max-2, totalK, start_ind, end_ind))

        ## Give the models to the model server. (Note: We used to pass them to workers via temp files which works when you create new
        ## workers but is harder when you need to coordinate timing of reading new models in every pass)

        t0 = time.time()
        if finite:
            DistributedModelCompiler.DistributedModelCompiler(depth, workDistributer, gpu, limit_depth=init_depth).compile_and_store_models(models, working_dir)
#            FullDepthCompiler.FullDepthCompiler(depth).compile_and_store_models(models, working_dir)
        else:
            NoopCompiler.NoopCompiler().compile_and_store_models(models, working_dir)
        t1 = time.time()
        workDistributer.submitSentenceJobs(start_ind, end_ind)

        ## Wait for server to finish distributing sentences for this iteration:
        t2 = time.time()

        logging.info("Sampling time of iteration %d: Model compilation: %d s; Sentence sampling: %d s" % (iter, t1-t0, t2-t1))


        ## Read the output and put it in a map -- use a map because sentences will
        ## finish asynchronously -- keep the sentence index so we can extract counts
        ## in whatever order we want (without aligning to sentences) but then order
        ## the samples so that when we print them out they align for inspection.
        t0 = time.time()
        num_processed = 0
        parses = workDistributer.get_parses()
        sample_map = dict()

        state_list = []
        state_indices = []
        for parse in parses:
            num_processed += 1
            # logging.info(''.join([x.str() for x in parse.state_list]))
            # logging.info(parse.success)
            if parse.success:
                try:
                    state_list.append(parse.state_list)
                    state_indices.append(ev_seqs[parse.index])
                    # logging.info('The state sequence is ' + ' '.join([str(indexer.getStateIndex(x.j, x.a, x.b, x.f, x.g)) for x in parse.state_list]))
                    # logging.info(' '.join([x.str() for x in parse.state_list]))

                    #increment_counts(parse.state_list, ev_seqs[parse.index], models)

                    # logging.info('Good parse:')
                    # logging.info(' '.join([x.str() for x in parse.state_list]))
                    # logging.info('The index is %d' % parse.index)
                except:
                    logging.error('This parse is bad:')
                    logging.error('The sentence is ' + ' '.join([str(x) for x in ev_seqs[parse.index]]))
                    logging.error('The state sequence is ' + ' '.join([str(indexer.getStateIndex(x.j, x.a, x.b, x.f, x.g)) for x in parse.state_list]))
                    logging.error(' '.join([x.str() for x in parse.state_list]))
                    logging.error('The index is %d' % parse.index)
                    raise
                sample.log_prob += parse.log_prob
            hid_seqs[parse.index] = parse.state_list
        pcfg_increment_counts(state_list, state_indices, models)

        if num_processed < (end_ind - start_ind):
            logging.warning("Didn't receive the correct number of parses at iteration %d" % iter)

        logging.info("Parsed %d sentences this batch -- now have %d parses" % (end_ind-start_ind, end_ind ) )
        max_state_check(hid_seqs, models, "parses")

        pos_counts = models.pos.pairCounts[:].sum()
        lex_counts = models.lex.pairCounts[:].sum()
        if not pos_counts == lex_counts:
            logging.warn("This iteration has %d pos counts for %d lex counts" % (pos_counts, lex_counts))

        if not pos_counts == num_tokens:
            logging.warn("This iteration has %d pos counts for %d tokens" % (pos_counts, num_tokens) )

        t1 = time.time()
        logging.info("Building counts tables took %d s" % (t1-t0))

        if iter >= burnin and (iter-burnin) % iters == 0:
            logging.info("Performed enough batches to collect a sample -- waiting for complete pass to finish")
            ready_for_sample = True

        if end_ind >= num_sents:
            if end_ind > num_sents:
                logging.warning("There are more parses than input sentences!")

            logging.info("Finished complete pass through data -- calling checkpoint function")
            sample.hid_seqs = hid_seqs
            checkpoint_function(sample)
            if ready_for_sample:
                logging.info("Collecting sample")
                #samples.append(sample)
                report_function(sample)
                logging.info(".\n")
                num_samples += 1
                ready_for_sample = False

            if split_merge:
                logging.info("Starting split/merge operation")
                models, sample = perform_split_merge_operation(models, sample, ev_seqs, params, iter, equal=True)
                hid_seqs = sample.hid_seqs
                max_state_check(hid_seqs, models, "split-merge")
                logging.info("Done with split/merge operation")
                report_function(sample)
                split_merge = False
                logging.debug("After split-merge the shape of root is %s and exp is %s \n" % (str(models.root[0].dist.shape), str(models.exp[0].dist.shape) ) )

            next_sample = Sample()
            #next_sample.hid_seqs = hid_seqs

            logging.info("Sampling hyperparameters")
            ## This is, e.g., where we might add categories to the a,b,g variables with
            ## stick-breaking. Without that, the values will stay what they were
            next_sample.alpha_f = sample.alpha_f
            next_sample.alpha_j = sample.alpha_j
            next_sample.alpha_a = models.A[0].alpha
            next_sample.alpha_b = models.B_J1[0].alpha
            next_sample.alpha_g = models.pos.alpha
            next_sample.alpha_h = models.lex.alpha
            
            next_sample.beta_f = sample.beta_f
            next_sample.beta_j = sample.beta_j
            next_sample.beta_a = models.A[0].beta
            next_sample.beta_b = models.B_J1[0].beta
            next_sample.beta_g = models.pos.beta
            next_sample.beta_h = models.lex.beta
            next_sample.gamma = sample.gamma
            next_sample.ev_seqs = ev_seqs

            prev_sample = sample
            sample = next_sample

        max_state_check(hid_seqs, models, "sampling hyperparameters")

        t0 = time.time()

        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        ## After stick-breaking we probably need to re-sample all the models:
        #remove_unused_variables(models, hid_seqs)
        if split_merge:
            resample_beta_g(models, sample.gamma)

        anneal_alphas = calc_anneal_alphas(models, iter, burnin, init_tempature, total_sent_lens)

        resample_all(models, sample, params, depth, anneal_alphas)

        ## Update sentence indices for next batch:
        if end_ind == len(ev_seqs):
            start_ind = 0
            end_ind = min(len(ev_seqs), batch_size)
        else:
            start_ind = end_ind
            end_ind = start_ind + min(len(ev_seqs), batch_size)
            if end_ind > num_sents:
                end_ind = num_sents

        sample.models = models

        if not finite and return_to_finite:
            finite = True
            return_to_finite = False

        pos_counts = models.pos.pairCounts[:].sum()
        lex_counts = models.lex.pairCounts[:].sum()
        logging.info("Have %d pos counts, %d lex counts after sample " % (pos_counts, lex_counts) )

        ## remove the counts from these sentences
        if batch_size < num_sents:
            logging.info("Decrementing counts for sentence indices %d-%d" % (start_ind, end_ind) )
            decrement_sentence_counts(hid_seqs, ev_seqs, models, start_ind, end_ind)
        else:
            logging.info("Resetting all counts to zero for next iteration")
            models.resetAll()
            pos_counts = models.pos.pairCounts[:].sum()
            lex_counts = models.lex.pairCounts[:].sum()
            assert pos_counts == 0 and lex_counts == 0

        iter += 1

    logging.debug("Ending sampling")
    workDistributer.stop()

    for cur_proc in range(0,num_procs):
        logging.info("Sending terminate signal to worker {} ...".format(cur_proc))
        inf_procs[cur_proc].terminate()

    for cur_proc in range(0,num_procs):
        logging.info("Waiting to join worker {} ...".format(cur_proc))
        inf_procs[cur_proc].join()
        inf_procs[cur_proc] = None

    return (samples, stats)

def max_state(hid_seqs):
    return max(a.g for b in hid_seqs for a in b)

def max_state_check(hid_seqs, models, location):
    ms = max_state(hid_seqs)
    logging.debug("max state in hid_seqs = " + str(ms) + ", pos dist size = " + str(models.pos.dist.shape[1]) + ", after " + location)
    assert ms <= models.pos.dist.shape[1]-2, \
        "Too many states in hid_seqs: max state of hid_seqs is %s, pos dist size is %s" % (ms, models.pos.dist.shape[1])
    if (ms < models.pos.dist.shape[1]-2):
      logging.warning("Too few states in hid_seqs: there is at least one state that is never used during inference.")

def remove_unused_variables(models, hid_seqs):
    ## Have to check if it's greater than 2 -- empty state (0) and final state (-1) are not supposed to have any counts:
    while sum(models.pos.pairCounts.sum(0)==0) > 2:
        pos = np.where(models.pos.pairCounts.sum(0)==0)[0][1]
        remove_pos_from_models(models, pos)
        remove_pos_from_hid_seqs(hid_seqs, pos)
    max_state_check(hid_seqs, models, "removing unused variables")

def remove_pos_from_hid_seqs(hid_seqs, pos):
    for a in hid_seqs:
        for state in a:
            assert state.g != pos, "Removed POS still exists in hid_seqs!"
            if state.g > pos:
                state.g = state.g - 1

def remove_pos_from_models(models, pos):
    ## delete the 1st dimension from the fork distribution:
    depth = len(models.fork)
    for d in range(depth):
        models.fork[d].pairCounts = np.delete(models.fork[d].pairCounts, pos, 1)
        models.fork[d].dist = np.delete(models.fork[d].dist, pos, 1)

        models.trans[d].pairCounts = np.delete(models.trans[d].pairCounts, pos, 1)
        models.trans[d].dist = np.delete(models.trans[d].dist, pos, 1)

        models.root[d].pairCounts = np.delete(models.root[d].pairCounts, pos, 1)
        models.root[d].dist = np.delete(models.root[d].dist, pos, 1)

        models.cont[d].pairCounts = np.delete(models.cont[d].pairCounts, pos, 1)
        models.cont[d].dist = np.delete(models.cont[d].dist, pos, 1)

        models.exp[d].pairCounts = np.delete(models.exp[d].pairCounts, pos, 0)
        models.exp[d].dist = np.delete(models.exp[d].dist, pos, 0)

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

    a_max = models.root[0].dist.shape[-1]
    b_max = models.cont[0].dist.shape[-1]

    ## Break the a stick (stored in root by convention -- TODO --  move this out to its own class later)
    break_beta_stick(models.root[0], sample.gamma)
    models.act[0].beta = models.root[0].beta

    ## Add a column to each of the out distributions (ACT and ROOT)
    add_model_column(models.root[0])
    add_model_column(models.act[0])

    ## For boolean variables can't do the simple row add:
    models.reduce[0].pairCounts = np.append(models.reduce[0].pairCounts, np.zeros((1,b_max,2)), 0)
    new_dist = np.log10(np.zeros((1,b_max,2)) + 0.5)
    models.reduce[0].dist = np.append(models.reduce[0].dist, new_dist, 0)

    ## Add row to the 1st dimension of act model:
    models.act[0].pairCounts = np.append(models.act[0].pairCounts, np.zeros((1, b_max, a_max+1)), 0)
    models.act[0].dist = np.append(models.act[0].dist, np.zeros((1, b_max, a_max+1)), 0)
    models.act[0].dist[a_max,...] = sampler.sampleDirichlet(models.act[0].pairCounts[a_max,...], models.act[0].alpha * models.act[0].beta)

    ## Add row to the 2nd dimension of exp model:
    models.exp[0].pairCounts = np.append(models.exp[0].pairCounts, np.zeros((g_max, 1, b_max)), 1)
    models.exp[0].dist = np.append(models.exp[0].dist, np.zeros((g_max, 1, b_max)), 1)
    models.exp[0].dist[:, a_max, :] = sampler.sampleDirichlet(models.exp[0].pairCounts[:, a_max, :], sample.alpha_b * sample.beta_b)

    ## Add row to the 2nd dimension of next model:
    models.next[0].pairCounts = np.append(models.next[0].pairCounts, np.zeros((b_max, 1, b_max)), 1)
    models.next[0].dist = np.append(models.next[0].dist, np.zeros((b_max, 1, b_max)), 1)
    models.next[0].dist[:, a_max, :] = sampler.sampleDirichlet(models.next[0].pairCounts[:, a_max, :], sample.alpha_b * sample.beta_b)

    ## Add a row to both dimensions of the model input
    ## need to modify shape to do appends so we'll make it a list from a tuple:
    ## Resize the distributions in the same way but we'll also need to initialize
    ## them to non-zero values
    models.start[0].pairCounts = np.append(models.start[0].pairCounts, np.zeros((1,a_max,b_max)), 0)
    models.start[0].dist = np.append(models.start[0].dist, np.zeros((1,a_max,b_max)), 0)
    models.start[0].dist[a_max,...] = sampler.sampleDirichlet(models.start[0].pairCounts[a_max,...], sample.alpha_b * sample.beta_b)

    models.start[0].pairCounts = np.append(models.start[0].pairCounts, np.zeros((a_max+1,1,b_max)), 1)
    models.start[0].dist = np.append(models.start[0].dist, np.zeros((a_max+1,1,b_max)), 1)
    models.start[0].dist[:,a_max,:] = sampler.sampleDirichlet(models.start[0].pairCounts[:,a_max,:], sample.alpha_b * sample.beta_b)

def break_b_stick(models, sample, params):

    b_max = models.cont[0].dist.shape[-1]
    g_max = models.pos.dist.shape[-1]

    ## Keep the stick with cont and copy it over to beta
    break_beta_stick(models.cont[0], sample.gamma)
    models.start[0].beta = models.cont[0].beta
    models.exp[0].beta = models.cont[0].beta
    models.next[0].beta = models.cont[0].beta

    ## Add a column to both output distributions:
    add_model_column(models.cont[0])
    add_model_column(models.start[0])
    add_model_column(models.exp[0])
    add_model_column(models.next[0])

    ## Add a row to the POS output distribution which depends only on b:
    add_model_row_simple(models.pos, models.pos.alpha * models.pos.beta[1:])

    ## Boolean models:
    models.fork[0].pairCounts = np.append(models.fork[0].pairCounts, np.zeros((1,g_max,2)), 0)
    models.fork[0].dist = np.append(models.fork[0].dist, np.zeros((1,g_max,2)), 0)
    models.fork[0].dist[b_max,...] = sampler.sampleBernoulli(models.fork[0].pairCounts[b_max,...], sample.alpha_f * sample.beta_f)

    models.trans[0].pairCounts = np.append(models.trans[0].pairCounts, np.zeros((1,g_max,2)), 0)
    models.trans[0].dist = np.append(models.trans[0].dist, np.zeros((1,g_max,2)), 0)
    models.trans[0].dist[b_max,...] = sampler.sampleBernoulli(models.trans[0].pairCounts[b_max,...], sample.alpha_j * sample.beta_j)

    models.reduce[0].pairCounts = np.append(models.reduce[0].pairCounts, np.zeros((a_max,1,2)), 1)
    models.reduce[0].dist = np.append(models.reduce[0].dist, np.zeros((a_max,1,2)), 1)
    models.reduce[0].dist[:,b_max,:] = sampler.sampleBernoulli(models.reduce[0].pairCounts[:,b_max,:], sample.alpha_j * sample.beta_j)

    ## Active models:
    models.act[0].pairCounts = np.append(models.act[0].pairCounts, np.zeros((a_max, 1, a_max)), 1)
    models.act[0].dist = np.append(models.act[0].dist, np.zeros((a_max, 1, a_max)), 1)
    models.act[0].dist[:,b_max,:] = sampler.sampleDirichlet(models.act[0].pairCounts[:,b_max,:], models.act[0].alpha * models.act[0].beta)

    models.root[0].pairCounts = np.append(models.root[0].pairCounts, np.zeros((1, g_max, a_max)), 0)
    models.root[0].dist = np.append(models.root[0].dist, np.zeros((1, g_max, a_max)), 0)
    models.root[0].dist[b_max,:,:] = sampler.sampleDirichlet(models.root[0].pairCounts[b_max,:,:], models.root[0].alpha * models.root[0].beta)

    ## Awaited models with b as condition
    models.cont[0].pairCounts = np.append(models.cont[0].pairCounts, np.zeros((1,g_max,b_max+1)), 0)
    models.cont[0].dist = np.append(models.cont[0].dist, np.zeros((1,g_max,b_max+1)),0)
    models.cont[0].dist[b_max,...] = sampler.sampleDirichlet(models.cont[0].pairCounts[b_max,...], models.cont[0].alpha * models.cont[0].beta)

    models.next[0].pairCounts = np.append(models.next[0].pairCounts, np.zeros((1,a_max,b_max+1)), 0)
    models.next[0].dist = np.append(models.next[0].dist, np.zeros((1,a_max,b_max+1)), 0)
    models.next[0].dist[b_max,...] = sampler.sampleDirichlet(models.next[0].pairCounts[b_max,...], models.next[0].alpha * models.next[0].beta)

def break_g_stick(models, sample, params):

    b_max = models.cont[0].dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    depth = len(models.fork)

    ## Resample beta when the stick is broken:
    break_beta_stick(models.pos, sample.gamma)

    if models.pos.beta[-1] == 0.0:
        logging.error("This shouldn't be 0!")

    ## Add a column to the distribution that outputs POS tags:
    add_model_column(models.pos)

    ## Add a row to the lexical distribution for this new POS tag:
    add_model_row_simple(models.lex, params['h'][0,1:])

    for d in range(0, depth):
        models.fork[d].pairCounts = np.append(models.fork[d].pairCounts, np.zeros((b_max,1,2)), 1)
        models.fork[d].dist = np.append(models.fork[d].dist, np.zeros((b_max,1,2)), 1)
        models.fork[d].dist[:,g_max,:] = sampler.sampleBernoulli(models.fork[d].pairCounts[:,g_max,:], sample.alpha_f * sample.beta_f)

        models.trans[d].pairCounts = np.append(models.trans[d].pairCounts, np.zeros((b_max,1,2)), 1)
        models.trans[d].dist = np.append(models.trans[d].dist, np.zeros((b_max,1,2)), 1)
        models.trans[d].dist[:,g_max,:] = sampler.sampleBernoulli(models.trans[d].pairCounts[:,g_max,:], sample.alpha_f * sample.beta_f)

        ## One active model uses g as a condition:
        models.root[d].pairCounts = np.append(models.root[d].pairCounts, np.zeros((b_max,1,a_max)), 1)
        models.root[d].dist = np.append(models.root[d].dist, np.zeros((b_max,1,a_max)), 1)
        models.root[d].dist[:,g_max,:] = sampler.sampleDirichlet(models.root[d].pairCounts[:,g_max,:], models.root[d].alpha * models.root[d].beta)

        ## Two awaited models use g as a condition:
        models.cont[d].pairCounts = np.append(models.cont[d].pairCounts, np.zeros((b_max,1,b_max)), 1)
        models.cont[d].dist = np.append(models.cont[d].dist, np.zeros((b_max,1,b_max)), 1)
        models.cont[d].dist[:,g_max,:] = sampler.sampleDirichlet(models.cont[d].pairCounts[:,g_max,:], models.cont[d].alpha * models.cont[d].beta)

        models.exp[d].pairCounts = np.append(models.exp[d].pairCounts, np.zeros((1, a_max, b_max)), 0)
        models.exp[d].dist = np.append(models.exp[d].dist, np.zeros((1, a_max, b_max)), 0)
        models.exp[d].dist[g_max,...] = sampler.sampleDirichlet(models.exp[d].pairCounts[g_max,...], models.exp[d].alpha * models.exp[d].beta)

def resample_beta_g(models, gamma):
    logging.info("Resampling beta g")
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
    #logging.info("New beta value is %s" % model.pos.beta)

def initialize_models(models, max_output, params, corpus_shape, depth, a_max, b_max, g_max):
    ## F model:
    models.F = [None] * depth
    ## J models:
    models.J = [None] * depth
    # models.reduce = [None] * depth
    ## Active models:
    models.A = [None] * depth
    # models.root = [None] * depth
    ## Reduce models:
    models.B_J1 = [None] * depth
    models.B_J0 = [None] * depth
    # models.next = [None] * depth
    # models.start = [None] * depth

    for d in range(0, depth):
        ## One fork model:
        models.F[d] = Model((b_max, 2), alpha=float(params.get('alphaf')), name="Fork"+str(d))
        models.F[d].beta = np.ones(2)

        ## Two join models:
        models.J[d] = Model((b_max, g_max, 2), alpha=float(params.get('alphaj')), name="Join"+str(d))
        models.J[d].beta = np.ones(2)
        
        # models.reduce[d] = Model((a_max, b_max, 2), alpha=float(params.get('alphaj')), name="J|F0_"+str(d))
        # models.reduce[d].beta = np.ones(2) / 2

        ## TODO -- set d > 0 beta to the value of the model at d (can probably do this later)
        ## One active model:
        models.A[d] = Model((a_max, b_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape, name="Act"+str(d))
        models.A[d].beta = np.ones(a_max)
        
        # models.root[d] = Model((b_max, g_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape, name="A|10_"+str(d))
        # models.root[d].beta = np.ones(a_max) / a_max

        ## four awaited models:
        models.B_J1[d] = Model((b_max, g_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|J1_"+str(d))
        models.B_J1[d].beta = np.ones(b_max)
        
        models.B_J0[d] = Model((g_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|J0_"+str(d))
        models.B_J0[d].beta = models.B_J1[d].beta
        #
        # models.next[d] = Model((a_max, b_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|01_"+str(d))
        # models.next[d].beta = models.cont[d].beta
        #
        # models.start[d] = Model((a_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|00_"+str(d))
        # models.start[d].beta = models.cont[d].beta


    ## one pos model:
    models.pos = Model((b_max, g_max), alpha=float(params.get('alphag')), corpus_shape=corpus_shape, name="POS")
    models.pos.beta = np.ones(g_max)

    ## one lex model:
    models.lex = Model((g_max, max_output+1), alpha=float(params.get('alphah')), name="Lex")
    models.lex.beta = np.ones(max_output+1)

    models.append(models.F)
    models.append(models.J)
    # models.append(models.reduce)
    models.append(models.A)
    # models.append(models.root)
    models.append(models.B_J1)
    # models.append(models.start)
    models.append(models.B_J0)
    # models.append(models.next)
    models.append(models.pos)
    models.append(models.lex)

    return models

# In the case that we are initializing this run with the output of a Previous
# run, it is pretty simple. We just need a lot of checks to make sure the inputs
# are actually compatible; depth of input can be <= but not > then specified
# max depth. Number of sentences and length of each sentence must be the same.
def initialize_and_load_state(ev_seqs, models, max_depth, init_seqs):
    input_depth = len(init_seqs[0][0].a)
    if input_depth > max_depth:
        logging.error("Sequence used for initialization has greater depth than max depth!")
        raise Exception

    if len(ev_seqs) != len(init_seqs):
        logging.error("Initialization sequence length %d is different than input sequence length %d." % (len(init_seqs), len(ev_seqs)))
        raise Exception

    state_seqs = list()
    for sent_index,sent in enumerate(ev_seqs):
        hid_seq = list()
        if len(sent) != len(init_seqs[sent_index]):
            logging.error("Sentence %d has length %d in init but length %d in input." % (sent_index, len(init_seqs[sent_index], len(sent))))
            raise Exception

        for index,word in enumerate(sent):
            state = State.State(max_depth)
            state.f = init_seqs[sent_index][index].f
            state.j = init_seqs[sent_index][index].j
            for d in range(0,input_depth):
                state.a[d] = init_seqs[sent_index][index].a[d]
                state.b[d] = init_seqs[sent_index][index].b[d]
            state.g = init_seqs[sent_index][index].g

            hid_seq.append(state)
        state_seqs.append(hid_seq)
        pcfg_increment_counts(hid_seq, sent, models)
        #increment_counts(hid_seq, sent, models)
    return state_seqs

# Randomly initialize all the values for the hidden variables in the
# sequence. Obeys constraints (e.g., when f=1,j=1 a=a_{t-1}) but otherwise
# samples randomly.
RANDOM_INIT=0
RB_INIT=1
LB_INIT=2

def initialize_state(ev_seqs, models, max_depth, gold_seqs=None, strategy=RANDOM_INIT):
    global a_max, b_max, g_max

    max_gold_tags = max([max(el) for el in gold_seqs.values()]) if (not gold_seqs == None and not len(gold_seqs)==0) else 0
    if not gold_seqs == None and max_gold_tags > g_max:
        logging.warning("There are more gold tag types (%d) than the number of initial pos tag states (%d). Will randomly initialize any gold tags that are out of bounds" % (max_gold_tags, g_max))

    state_seqs = list()
    for sent_index,sent in enumerate(ev_seqs):
        hid_seq = list()
        if not gold_seqs == None and sent_index in gold_seqs.keys():
          gold_tags=gold_seqs[sent_index]
        else:
          gold_tags=None
        for index,word in enumerate(sent):
            state = State.State(max_depth)
            ## special case for first word
            if index == 0:
                state.f = 1
                state.j = 0
                state.a[0] = 0
                state.b[0] = 0
            else:
                prev_depth = prev_state.max_awa_depth()
                # if index == 1:
                #     state.f = 1
                #     state.j = 0
                if strategy == RANDOM_INIT:
                    if np.random.random() >= 0.5:
                        state.f = 1
                    else:
                        state.f = 0
                    ## j is deterministic in the middle of the sentence
                    if prev_depth+1 == max_depth and state.f == 1:
                        ## If we are at max depth and we forked, we must reduce
                        state.j = 1
                    elif prev_depth <= 0 and state.f == 0:
                        ## If we are at 0 and we did not fork, we must not reduce
                        state.j = 0
                    else:
                        state.j = state.f

                elif strategy == RB_INIT:
                    state.f = 1
                    state.j = 1
                elif strategy == LB_INIT:
                    state.f = 0
                    state.j = 0
                else:
                    logging.error("Unknown initialization strategy %d" % (strategy))
                    raise Exception
                if index == 1:
                    state.j = 0

                cur_depth = prev_depth - state.j
                if prev_depth == -1:
                    state.a[cur_depth] = np.random.randint(1, a_max-1)
                elif state.f == 1 and state.j == 1:
                    state.a[0:cur_depth] = prev_state.a[0:cur_depth]
                    state.b[0:cur_depth] = prev_state.b[0:cur_depth]
                    state.a[cur_depth] = prev_state.a[cur_depth]
                elif state.f == 0 and state.j == 0:
                    state.a[0:cur_depth] = prev_state.a[0:cur_depth]
                    state.b[0:cur_depth] = prev_state.b[0:cur_depth]
                    state.a[cur_depth] = np.random.randint(1,a_max-1)
                elif state.f == 1 and state.j == 0:
                    state.a[0:cur_depth] = prev_state.a[0:cur_depth]
                    state.b[0:cur_depth] = prev_state.b[0:cur_depth]
                    state.a[cur_depth] = np.random.randint(1,a_max-1)
                elif state.f == 0 and state.j == 1:
                    state.a[0:cur_depth] = prev_state.a[0:cur_depth]
                    state.b[0:cur_depth] = prev_state.b[0:cur_depth]
                    state.a[cur_depth] = prev_state.a[cur_depth]
                else:
                    raise Exception("Encountered unexpected condition in state initialization! f=%s, j=%s" % (state.f, state.j) )

                state.b[cur_depth] = np.random.randint(1,b_max-1)

            if gold_tags and gold_tags[index] < g_max:
              state.g = gold_tags[index]
            else:
              state.g = np.random.randint(1,g_max-1)

            prev_state = state

            hid_seq.append(state)

        pcfg_increment_counts(hid_seq, sent, models)
        #increment_counts(hid_seq, sent, models)
        state_seqs.append(hid_seq)

    return state_seqs

def collect_trans_probs(hid_seqs, models, start_ind, end_ind):
    if end_ind-1 >= len(hid_seqs):
        logging.error("Index of end_ind=%d passed in to sequence of length %d" % (end_ind, len(hid_seqs) ) )

    for sent_index in range(start_ind, end_ind):
        hid_seq = hid_seqs[sent_index]
        if hid_seq == None:
            logging.warning("collect_trans_probs() found a missing parse %d - skipping sentence." % (sent_index) )
            continue

        ## for every state transition in the sentence increment the count
        ## for the condition and for the output
        for index, state in enumerate(hid_seq):
            d = 0

            if index != 0:
                if index == 1:
                    models.root[0].trans_prob[sent_index, index] = models.root[0].dist[0, prev_state.g, state.a[0]]
                    models.exp[0].trans_prob[sent_index, index] = models.exp[0].dist[prev_state.g, state.a[0], state.b[0]]
                else:
                    ## Fork and join don't have trans_probs because they are finite:
                    if state.f == 0 and state.j == 0:
                        models.act[d].trans_prob[sent_index, index] = models.root[d].trans_prob[sent_index, index] = models.act[d].dist[ prev_state.a[d], 0, state.a[d] ]
                        models.start[d].trans_prob[sent_index, index] = models.cont[d].trans_prob[sent_index, index] = models.exp[d].trans_prob[sent_index, index] = models.start[d].dist[ prev_state.a[d], state.a[d], state.b[d] ]
                    elif state.f == 1 and state.j == 1:
                        models.act[d].trans_prob[sent_index, index] = models.root[d].trans_prob[sent_index, index] = 0
                        models.cont[d].trans_prob[sent_index, index] = models.start[d].trans_prob[sent_index, index] = models.exp[d].trans_prob[sent_index, index] = models.cont[d].dist[ prev_state.b[d], prev_state.g, state.b[d] ]
                    elif state.f == 1 and state.j == 0:
                        models.root[d].trans_prob[sent_index, index] = models.act[d].trans_prob[sent_index, index]  = models.root[d].dist[ 0, prev_state.g, state.a[d] ]
                        models.cont[d].trans_prob[sent_index, index] = models.start[d].trans_prob[sent_index, index] = models.exp[d].trans_prob[sent_index, index] = models.exp[0].dist[prev_state.g, state.a[d], state.b[d] ]
                    elif state.f == 0 and state.j == 1:
                        models.act[d-1].trans_prob[sent_index, index] = models.root[d-1].trans_prob[sent_index, index] = 0
                        models.cont[d].trans_prob[sent_index, index] = models.start[d].trans_prob[sent_index, index] = models.cont[d-1].dist[ prev_state.b[d-1], prev_state.g, state.b[d] ]
            models.pos.trans_prob[sent_index, index] = models.pos.dist[state.b[0], state.g]
            prev_state = state

def increment_counts(hid_seq, sent, models, inc=1):
    ## for every state transition in the sentence increment the count
    ## for the condition and for the output

    # Create start state
    max_depth = len(hid_seq[0].a)
    prev_state = State.State(max_depth)
    prev_state.f = 0
    prev_state.j = 0
    prev_state.a = np.asarray([0]*max_depth)
    prev_state.b = np.asarray([0]*max_depth)
    prev_state.g = 0
    depth = -1

    # Create end state
    EOS = State.State(max_depth)
    EOS.f = 0
    EOS.j = 1
    EOS.a = np.asarray([0]*max_depth)
    EOS.b = np.asarray([0]*max_depth)
    EOS.g = 0

    # Append end state
    sent = sent[:] + [0]
    hid_seq = hid_seq[:] + [EOS]
    logging.debug('now working on state sequence: '+'  '.join(some_state.unfiltered_str() for some_state in hid_seq))
    for index,word in enumerate(sent):
        state = hid_seq[index]
        logging.debug("state {}: {}; word {}; prev_state {}".format(index, state.unfiltered_str(), word, prev_state.unfiltered_str()))
        # Populate previous state conditional dependencies
        prev_g = prev_state.g
        if depth == -1:
            prev_a = 0
            prev_b = 0
            prev_b_above = 0
        else:
            prev_a = prev_state.a[depth]
            prev_b = prev_state.b[depth]
            if depth == 0:
                prev_b_above = 0
            else:
                prev_b_above = prev_state.b[depth-1]
        prev_f = prev_state.f

        # Count join decision
        if index != 0:#  and index != len(sent) - 1:
            if prev_f == 0:
                if depth >= 0 and (prev_a == 0 and prev_b_above == 0):
                    print('Collision check -- J model at depth >=0 has same conditions as at depth -1.')
                ## Final state is deterministic, don't include counts from final decisions:
                # if word != 0:
                models.J[max(0,depth)].count((prev_a, prev_b_above), state.j, inc)
                logging.debug("J model inc count: {} {} -> {} at {}".format(prev_a, prev_b_above, state.j, max(0,depth)))
            elif prev_f == 1:
                if depth >= 0 and (prev_b == 0 and prev_g == 0):
                    print('Collision check -- J model at depth >=0 has same conditions as at depth -1.')
                if depth + prev_f < max_depth:
                    models.J[max(0, depth+1)].count((prev_g, prev_b), state.j, inc)
                    logging.debug("J model inc count: {} {} -> {} at {}".format(prev_g, prev_b, state.j, max(0, depth+1)))
            else:
                raise Exception("Unallowed value (%s) of the fork variable!" %state.f)

        # Populate current state conditional dependencies
        cur_depth = depth + prev_f - state.j
        cur_g = state.g
        if cur_depth == -1:
            cur_a = 0
            cur_b = 0
            cur_b_above = 0
        else:
            cur_a = state.a[cur_depth]
            cur_b = state.b[cur_depth]
            if cur_depth == 0:
                cur_b_above = 0
            else:
                cur_b_above = state.b[cur_depth-1]

        ## Count A & B
        if index != 0 and word != 0:
            if prev_f == 0 and state.j == 0:
                assert depth >= 0 or index == 0, "Found a non-initial -/- decision at depth -1 (should not be possible)."
                if depth >= 0 and (prev_a == 0 and prev_b_above == 0):
                    print('Collision check -- A model at depth >=0 has same conditions as at depth -1.')
                models.A[max(0,depth)].count((prev_b_above, prev_a), cur_a, inc)
                logging.debug("A model inc count: {} {} -> {} at {}".format(prev_b_above, prev_a, cur_a, max(0,depth)))
                if depth >= 0 and (prev_a == 0 and cur_a == 0):
                    print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
                models.B_J0[max(0,depth)].count((cur_a, prev_a), cur_b, inc)
                logging.debug("B_J0 model inc count: {} {} -> {} at {}".format(cur_a, prev_a, cur_b, max(0,depth)))

            elif prev_f == 1 and state.j == 1:
                assert depth >= 0 or (len(sent) == 2 and index == 1), "Found an illegal +/+ transition at depth -1. %s %s" %(sent, index)
                if depth >= 0 and (prev_b == 0 and prev_g == 0):
                    print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
                models.B_J1[max(0,depth)].count((prev_b, prev_g), cur_b, inc)
                logging.debug("B_J1 model inc count: {} {} -> {} at {}".format(prev_b, prev_g, cur_b, max(0, depth)))

            elif prev_f == 1 and state.j == 0:
                assert depth <= max_depth, "Found a +/- decision at the maximum depth level."
                if depth >= 0 and (prev_b == 0 and prev_g == 0):
                    print('Collision check -- A model at depth >=0 has same conditions as at depth -1.')
                models.A[depth+1].count((prev_b, prev_g), cur_a, inc)
                logging.debug("A model inc count: {} {} -> {} at {}".format(prev_b, prev_g, cur_a, depth+1))
                if depth >= 0 and (prev_g == 0 and cur_a == 0):
                    print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
                models.B_J0[depth+1].count((cur_a, prev_g), cur_b, inc)
                logging.debug("B_J0 model inc count: {} {} -> {} at {}".format(cur_a, prev_g, cur_b, depth+1))

            elif prev_f == 0 and state.j == 1:
                assert depth > 0 or index == len(sent) - 1, "Found a -/+ decision at depth 0 prior to sentence end."
                if depth >= 0 and (prev_a == 0 and prev_b_above == 0):
                    print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
                models.B_J1[max(0,depth-1)].count(( prev_b_above, prev_a), cur_b, inc)
                logging.debug("B_J1 model inc count: {} {} -> {} at {}".format(prev_b_above, prev_a, cur_b, max(0, depth-1)))
            else:
                raise Exception("Unallowed value of f=%d and j=%d, index=%d" % (state.f, state.j, index) )

        # Count fork decision
        if cur_depth >= 0 and (cur_b == 0 and cur_g == 0) and index != 0:
            print('Collision check -- F model at depth >=0 has same conditions as at depth -1.')
        ## Final state is deterministic, don't include counts from final decisions:
        if word != 0 and index != 0:# and index != len(sent) - 2:
            models.F[max(0,depth)].count(cur_b, state.f, inc)
            logging.debug("F model inc count: {} -> {} at {}".format(cur_b, state.f, max(0, depth)))

        ## Count G
        if word != 0 and state.f != 0:
            models.pos.count(cur_b, cur_g, inc)
            logging.debug("G model inc count: {} -> {}".format(cur_b, cur_g))
        ## Count w
        if word != 0:
            models.lex.count(cur_g, word, inc)
            logging.debug("W model inc count: {} -> {}".format(cur_g, word))

        depth = state.max_awa_depth()
        prev_state = state

def decrement_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    for ind in range(start_ind, end_ind):
        pcfg_increment_counts(hid_seqs[ind], sents[ind], models, -1)
        #increment_counts(hid_seqs[ind], sents[ind], models, -1)

def increment_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    for ind in range(start_ind, end_ind):
        pcfg_increment_counts(hid_seqs[ind], sents[ind], models, 1)
        #increment_counts(hid_seqs[ind], sents[ind], models, 1)

def handle_sigint(signum, frame, workers):
    logging.info("Master received quit signal... will terminate after cleaning up.")
    for ind,worker in enumerate(workers):
        logging.info("Terminating worker %d" % (ind) )
        worker.terminate()
        logging.info("Joining worker %d" % (ind) )
        worker.join()
    sys.exit(0)

def getGmax():
    global g_max
    return g_max

def getAmax():
    global a_max
    return a_max

def getBmax():
    global b_max
    return b_max

def resample_all(models, sample, params, depth, anneal_alphas=0):
    ## Sample distributions for all the model params and emissions params
    ## TODO -- make the Models class do this in a resample_all() method
    if anneal_alphas == 0:
        a_base =  sample.alpha_a * sample.beta_a
        b_base = sample.alpha_b * sample.beta_b
        f_base = sample.alpha_f * sample.beta_f
        j_base = sample.alpha_j * sample.beta_j
        g_base = sample.alpha_g * sample.beta_g
        h_base = sample.alpha_h * sample.beta_h
    elif isinstance(anneal_alphas, dict):
        a_base = (sample.alpha_a + anneal_alphas['A']) * sample.beta_a
        b_base = (sample.alpha_b + anneal_alphas['B_J0'])* sample.beta_b
        f_base = (sample.alpha_f + anneal_alphas['F'])* sample.beta_f
        j_base = (sample.alpha_j + anneal_alphas['J']) * sample.beta_j
        g_base = (sample.alpha_g + anneal_alphas['pos']) * sample.beta_g
        h_base = (sample.alpha_h + anneal_alphas['lex']) * sample.beta_h
    else:
        raise Exception("Anneal alphas are neither 0 nor a dictionary!")

    # Resample lex and make sure the null tag can only generate the null word
    models.lex.sampleDirichlet(h_base)
    models.lex.dist[0,0] = 0.0
    models.lex.dist[0,1:].fill(-np.inf)

    # Resample pos
    print("model P: P|B")
    print(models.pos.pairCounts)
    models.pos.sampleDirichlet(g_base)
    if np.argwhere(np.isnan(models.pos.dist)).size > 0:
        logging.error("Resampling the pos dist resulted in a nan")

    logging.debug('printing out the pair counts for the models')
    for d in range(depth-1, -1, -1):
        print('depth {} model B J1: B| (B A) or (B P)'.format(d))
        print(models.B_J1[d].pairCounts)
        models.B_J1[d].sampleDirichlet(b_base) # if d == 0 else b_base + models.B_J1[d-1].pairCounts * sample.alpha_b)
        print('depth {} model B J0: B| (A A) or (A P)'.format(d))
        print(models.B_J0[d].pairCounts)
        models.B_J0[d].sampleDirichlet(b_base) # if d == 0 else b_base + models.B_J0[d-1].pairCounts * sample.alpha_b)
        # models.cont[d].sampleDirichlet(b_base if d == 0 else b_base + models.cont[d-1].pairCounts * sample.alpha_b)
        # models.next[d].sampleDirichlet(b_base if d == 0 else b_base + models.next[d-1].pairCounts * sample.alpha_b)
        print("depth {} model A: A|(B A) or (B P)".format(d))
        print(models.A[d].pairCounts)
        models.A[d].sampleDirichlet(a_base) # if d == 0 else a_base + models.A[d-1].pairCounts * sample.alpha_a)
        # models.root[d].sampleDirichlet(a_base if d == 0 else a_base + models.root[d-1].pairCounts * sample.alpha_a)
        # models.reduce[d].sampleDirichlet(j_base if d == 0 else j_base + models.reduce[d-1].pairCounts * sample.alpha_j)
        print("depth {} model J: J|(A B) or (P B)".format(d))
        print(models.J[d].pairCounts)
        models.J[d].sampleDirichlet(j_base) # if d == 0 else j_base + models.J[d-1].pairCounts * sample.alpha_j)
        print("depth {} model F: F|B".format(d))
        print(models.F[d].pairCounts)
        models.F[d].sampleDirichlet(f_base) # if d == 0 else f_base + models.F[d-1].pairCounts * sample.alpha_f)
