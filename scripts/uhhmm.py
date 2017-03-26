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

    sent_lens = list(map(len, ev_seqs))
    maxLen = max(map(len, ev_seqs))
    max_output = max(map(max, ev_seqs))
    num_sents = len(ev_seqs)
    num_tokens = np.sum(sent_lens)

    num_samples = 0

    ## Set debug first so we can use it during config setting:
    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile','')
    logging.basicConfig(level=getattr(logging, debug),filename=logfile)

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
    inc = params.get('inc', 1)
    infinite_sample_prob = float(params.get('infinite_prob', 0.0))
    batch_size = min(num_sents, int(params.get('batch_size', num_sents)))
    gpu = bool(int(params.get('gpu', 0)))
    gpu_batch_size = min(num_sents, int(params.get('gpu_batch_size', 32 if gpu == 1 else 1)))
    from_global_counts = bool(int(params.get('from_global_counts', 0)))
    decay = float(params.get('decay', 1.0))
    if gpu and num_gpu_workers < 1:
        logging.warn("Inconsistent config: gpu flag set with %d gpu workers; setting gpu=False" % (num_gpu_workers))
        gpu=False

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

    if pickle_file is None:
        ## Add 1 to every start value for "Null/start" state
        a_max = start_a+2
        b_max = start_b+2
        g_max = start_g+2

        models.initialize_as_fjabp(max_output, params, (len(ev_seqs), maxLen), depth, a_max, b_max, g_max)
        if input_seqs_file is None:
            hid_seqs = [None] * len(ev_seqs)
        else:
            logging.info("Trainer was initialized with input sequences from a previous run.")
            hid_seqs = initialize_and_load_state(ev_seqs, models, depth, uhhmm_io.read_input_states(input_seqs_file, depth))
            max_state_check(hid_seqs, models, "initialization")

        sample = Sample()
        sample.alpha_fj = models.fj[0].alpha
        sample.alpha_a = models.root[0].alpha
        sample.alpha_b = models.cont[0].alpha
        sample.alpha_g = models.pos.alpha
        sample.alpha_h = models.lex.alpha
        
        sample.beta_fj = models.fj[0].beta 
        sample.beta_a = models.root[0].beta 
        sample.beta_b = models.cont[0].beta
        sample.beta_g = models.pos.beta
        sample.beta_h = models.lex.beta
        sample.gamma = float(params.get('gamma'))
        sample.discount = float(params.get('discount'))
        sample.ev_seqs = ev_seqs

        sample.models = models
        models.resample_all(decay, init=True)
        iter = 0

    else:
        sample = uhhmm_io.read_serialized_sample(pickle_file)
        sample.log_prob = 0
        models = sample.models
        hid_seqs = sample.hid_seqs
        max_state_check(hid_seqs, models, "reading sample from pickle file")

        pos_counts = models.pos.pairCounts[:].sum()
        lex_counts = models.lex.pairCounts[:].sum()

        a_max = models.act[0].dist.shape[-1]
        b_max = models.cont[0].dist.shape[-1]
        g_max = models.pos.dist.shape[-1]

        models.resample_all(decay, from_global_counts)
        iter = sample.iter+1

    models.resetAll()

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
                models.break_g_stick(sample.gamma)
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

        for parse in parses:
            num_processed += 1
            # logging.info(''.join([x.str() for x in parse.state_list]))
            # logging.info(parse.success)
            if parse.success:
                try:

                    increment_counts(parse.state_list, ev_seqs[ parse.index ], models, inc)
                    # logging.info('Good parse:')
                    # logging.info(' '.join([x.str() for x in parse.state_list]))
                    # logging.info('The index is %d' % parse.index)
                except:
                    logging.error('This parse is bad:')
                    logging.error('The sentence is ' + ' '.join([str(x) for x in ev_seqs[parse.index]]))
                    logging.error('The state sequence is ' + ' '.join([str(indexer.getStateIndex(x.f, x.j, x.a, x.b, x.g)) for x in parse.state_list]))
                    logging.error(' '.join([x.str() for x in parse.state_list]))
                    logging.error('The index is %d' % parse.index)
                    raise(ValueError)
                sample.log_prob += parse.log_prob
            hid_seqs[parse.index] = parse.state_list

        models.increment_global_counts()

        if num_processed < (end_ind - start_ind):
            logging.warning("Didn't receive the correct number of parses at iteration %d" % iter)

        logging.info("Parsed %d sentences this batch -- now have %d parses" % (end_ind-start_ind, end_ind ) )
        max_state_check(hid_seqs, models, "parses")

        pos_counts = models.pos.pairCounts[:].sum() - num_sents
        lex_counts = models.lex.pairCounts[:].sum() - num_sents
        assert pos_counts == lex_counts

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
            next_sample.alpha_fj = models.fj[0].alpha
            next_sample.alpha_a = models.root[0].alpha
            next_sample.alpha_b = models.cont[0].alpha
            next_sample.alpha_g = models.pos.alpha
            next_sample.alpha_h = models.lex.alpha
            
            next_sample.beta_fj = models.fj[0].beta
            next_sample.beta_a = models.root[0].beta
            next_sample.beta_b = models.cont[0].beta
            next_sample.beta_g = models.pos.beta
            next_sample.beta_h = models.lex.beta
            next_sample.gamma = sample.gamma
            next_sample.ev_seqs = ev_seqs

            prev_sample = sample
            sample = next_sample

        max_state_check(hid_seqs, models, "sampling hyperparameters")

        t0 = time.time()

        ## Sample distributions for all the model params and emissions params
        ## After stick-breaking we probably need to re-sample all the models:
        #remove_unused_variables(models, hid_seqs)
        #resample_beta_g(models, sample.gamma)

        models.resample_all(decay, from_global_counts)

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

        pos_counts = models.pos.pairCounts[:].sum() - num_sents
        lex_counts = models.lex.pairCounts[:].sum() - num_sents
        logging.info("Have %d pos counts, %d lex counts after sample - should equal number of tokens %d" % (pos_counts, lex_counts, num_tokens) )

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
        models.remove_pos(pos)
        remove_pos_from_hid_seqs(hid_seqs, pos)
    max_state_check(hid_seqs, models, "removing unused variables")

def remove_pos_from_hid_seqs(hid_seqs, pos):
    for a in hid_seqs:
        for state in a:
            assert state.g != pos, "Removed POS still exists in hid_seqs!"
            if state.g > pos:
                state.g = state.g - 1

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
        increment_counts(hid_seq, sent, models, inc)
        models.increment_global_counts()

    return state_seqs

def collect_trans_probs(hid_seqs, models, start_ind, end_ind):
    if end_ind-1 >= len(hid_seqs):
        logging.error("Index of end_ind=%d passed in to sequence of length %d" % (end_ind, len(hid_seqs) ) )

    for sent_index in range(start_ind, end_ind):
        hid_seq = hid_seqs[sent_index]
        if hid_seq is None:
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
    depth = len(models.fj)

    # Initialize incrementer
    inc = str(inc)
    rand_inc = False
    if inc.startswith('rand'):
        rand_inc = True
        ceil = int(inc[4:])
    else:
        inc = int(inc)

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
    if len(sent) == 1:
        EOS = State.State(max_depth)
        EOS.f = 1
        EOS.j = 1
        EOS.a = np.asarray([0]*max_depth)
        EOS.b = np.asarray([0]*max_depth)
        EOS.g = 0
    else:
        EOS = State.State(max_depth)
        EOS.f = 0
        EOS.j = 1
        EOS.a = np.asarray([0]*max_depth)
        EOS.b = np.asarray([0]*max_depth)
        EOS.g = 0

    # Append end state
    sent = sent[:] + [0]
    hid_seq = hid_seq[:] + [EOS]

    for index,word in enumerate(sent):
        # Set incrementer
        if rand_inc:
            inc = np.random.randint(0,ceil) + 1

        state = hid_seq[index]

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

        # Count fork decision
        if depth >= 0 and (prev_a == 0 and prev_b == 0 and prev_b_above == 0 and prev_g == 0):
            print('Collision check -- FJ model at depth >=0 has same conditions as at depth -1.')
        models.fj[max(0,depth)].count((prev_a, prev_b, prev_b_above, prev_g), 2*state.f+state.j, inc)

        # Populate current state conditional dependencies
        cur_depth = depth + state.f - state.j
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
        if state.f == 0 and state.j == 0:
            assert depth >= 0 or index == 0, "Found a non-initial -/- decision at depth -1 (should not be possible)."
            if depth >= 0 and (prev_a == 0 and prev_b_above == 0):
                print('Collision check -- A model at depth >=0 has same conditions as at depth -1.')
            models.act[max(0,depth)].count((prev_a, prev_b_above), cur_a, inc)
            if depth >= 0 and (prev_a == 0 and cur_a == 0):
                print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
            models.start[max(0,depth)].count((prev_a, cur_a), cur_b, inc)

        elif state.f == 1 and state.j == 1:
            assert depth >= 0 or (len(sent) == 2 and index == 1), "Found an illegal +/+ transition at depth -1. %s %s" %(sent, index)
            if depth >= 0 and (prev_b == 0 and prev_g == 0):
                print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
            models.cont[max(0,depth)].count((prev_b, prev_g), cur_b, inc)

        elif state.f == 1 and state.j == 0:
            assert depth <= max_depth, "Found a +/- decision at the maximum depth level."
            if depth >= 0 and (prev_b == 0 and prev_g == 0):
                print('Collision check -- A model at depth >=0 has same conditions as at depth -1.')
            models.root[depth+1].count((prev_b, prev_g), cur_a, inc)
            if depth >= 0 and (prev_g == 0 and cur_a == 0):
                print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
            models.exp[depth+1].count((prev_g, cur_a), cur_b, inc)

        elif state.f == 0 and state.j == 1:
            assert depth > 0 or index == len(sent) - 1, "Found a -/+ decision at depth 0 prior to sentence end."
            if depth >= 0 and (prev_a == 0 and prev_b_above == 0):
                print('Collision check -- B model at depth >=0 has same conditions as at depth -1.')
            models.next[max(0,depth-1)].count((prev_a, prev_b_above), cur_b, inc)

        else:
            raise Exception("Unallowed value of f=%d and j=%d, index=%d" % (state.f, state.j, index) )

        ## Count G
        models.pos.count(cur_b, cur_g, inc)
        ## Count w
        models.lex.count(cur_g, word, inc)

        depth = state.max_awa_depth()
        prev_state = state

def decrement_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    for ind in range(start_ind, end_ind):
        increment_counts(hid_seqs[ind], sents[ind], models, -1)

def increment_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    for ind in range(start_ind, end_ind):
        increment_counts(hid_seqs[ind], sents[ind], models, 1)

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

