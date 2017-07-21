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
import os
from multiprocessing import Process, Queue, JoinableQueue
import zmq
from WorkDistributerServer import WorkDistributerServer
from PyzmqMessage import *
# from PyzmqWorker import PyzmqWorker
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
from pcfg_translator import *
import copy
from init_pcfg_strategies import *
from pcfg_model import PCFG_model
from gpu_parse import parse, compile_and_set_models, initialize_models, handle_sigint, PARSING_SIGNAL_FILE

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
def sample_beam(ev_seqs, params, report_function, checkpoint_function, working_dir, pickle_file=None, gold_seqs=None,
                input_seqs_file=None, word_dict_file=None):
    global start_abp
    start_abp = int(params.get('startabp'))

    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile', '')
    if logfile:
        logging.basicConfig(level=getattr(logging, debug), filename=logfile)
    else:
        logging.basicConfig(level=getattr(logging, debug), stream=sys.stdout)
    # validation settings
    validation = int(params.get("validation", 0))  # this is for validation with MH steps
    validation_no_mh = int(params.get("validation_no_mh", 0)) # this is for validation with no MH steps just for validation log prob graphing
    validation_length = int(params.get("validation_length", 1000)) # the last 1000 senteces are used as validation set by default
    if validation:
        logging.info('the validation set is the last {} sentences of all the data.'.format(validation_length))
        validation_length = -validation_length
    else:
        logging.info('validation is off.')
        validation_length = None

    maxLen = max(map(len, ev_seqs))
    max_output = max(map(max, ev_seqs))
    # if validation:
    #     ev_seqs, val_seqs = ev_seqs[:-validation_length], ev_seqs[-validation_length:]
    # sent_lens = list(map(len, ev_seqs))
    # total_sent_lens = sum(sent_lens)
    num_ev_sents = len(ev_seqs[:validation_length])
    # num_tokens = np.sum(sent_lens)

    num_samples = 0

    depth = int(params.get('depth', 1))
    init_depth = int(params.get('init_depth', depth))
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    max_samples = int(params.get('num_samples'))
    num_cpu_workers = int(params.get('cpu_workers', 0))
    num_procs = int(params.get('num_procs', -1))
    if num_procs != -1:
        num_cpu_workers = num_procs
        logging.warn(
            "num_procs config is deprecated for cpu_workers and gpu_workers configs. Treating num_procs as num_cpu_workers=%d" % (
            num_cpu_workers))

    num_gpu_workers = int(params.get('gpu_workers', 0))

    profile = bool(int(params.get('profile', 0)))
    finite = bool(int(params.get('finite', 0)))
    cluster_cmd = params.get('cluster_cmd', None)
    split_merge_iters = int(params.get('split_merge_iters', -1))
    infinite_sample_prob = float(params.get('infinite_prob', 0.0))
    batch_size = min(num_ev_sents, int(params.get('batch_size', num_ev_sents)))
    gpu = bool(int(params.get('gpu', 0)))
    gpu_batch_size = min(num_ev_sents, int(params.get('gpu_batch_size', 32 if gpu == 1 else 1)))
    if gpu and num_gpu_workers < 1 and num_cpu_workers > 0:
        logging.warn("Inconsistent config: gpu flag set with %d gpu workers; setting gpu=False" % (num_gpu_workers))
        gpu = False

    # these are annealing parameters
    init_anneal_alpha = float(params.get('init_anneal_alpha', 0))
    init_anneal_likelihood = float(params.get("init_anneal_likelihood", 0))
    final_anneal_likelihood = float(params.get("final_anneal_likelihood", 1))
    anneal_length = int(params.get("anneal_length", 0))
    anneal_likelihood_phase = int(params.get("anneal_likelihood_phase", anneal_length))
    random_restarts = int(params.get("random_restarts", 0))
    gold_pcfg_file = params.get("gold_pcfg_file", '')
    add_noise = int(params.get("add_noise", 0))
    noise_sigma = float(params.get('noise_sigma', 0))
    init_strategy = params.get("init_strategy", '')
    always_sample = int(params.get("always_sample", 0))
    gold_pos_dict_file = params.get("gold_pos_dict_file", '')
    MAP = int(params.get("MAP", 0))
    normalize_flag = int(params.get("normalize_flag", 1))
    alpha_pcfg_range = params.get('alpha_pcfg_range',
                                  [0.1, 1.0])  # a comma separated list of lower and upper bounds of alpha-pcfg
    init_alpha = float(params.get("init_alpha", 0.2))
    sample_alpha_flag = int(params.get("sample_alpha_flag", 0))

    # super cooling
    super_cooling = int(params.get("super_cooling", 0))
    super_cooling_length = int(params.get("super_cooling_length", 500))
    super_cooling_start_iter = int(params.get("super_cooling_start_iter", 2000))
    super_cooling_target_ac = int(params.get("super_cooling_target_ac", 10))

    # viterbi decoding
    viterbi = int(params.get("viterbi", 0))

    # main body
    if gold_pos_dict_file:
        gold_pos_dict = {}
        with open(gold_pos_dict_file) as g:
            for line in g:
                line = line.strip().split(' = ')
                gold_pos_dict[int(line[0])] = int(line[1])

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
    end_ind = min(num_ev_sents, batch_size)

    pcfg_model = PCFG_model(start_abp, max_output, log_dir=working_dir, word_dict_file = word_dict_file)
    pcfg_model.set_alpha(alpha_pcfg_range, alpha=init_alpha)

    logging.info("Initializing state")

    if pickle_file is None:
        ## Add 1 to every start value for "Null/start" state
        inflated_num_abp = start_abp + 2

        models = initialize_models(models, max_output, params, (len(ev_seqs), maxLen), depth, inflated_num_abp)
        if input_seqs_file is None:
            hid_seqs = [None] * len(ev_seqs)

        else:
            logging.info("Trainer was initialized with input sequences from a previous run.")
            hid_seqs = initialize_and_load_state(ev_seqs, models, depth,
                                                 uhhmm_io.read_input_states(input_seqs_file, depth))
            max_state_check(hid_seqs, models, "initialization")

        sample = Sample()
        sample.ev_seqs = ev_seqs
        pcfg_model.start_logging()
        # initialization: a few controls:
        if gold_pcfg_file:
            logging.info("Initializing the models with the gold PCFG file {}.".format(gold_pcfg_file))
            pcfg_replace_model(None, None, models, gold_pcfg_file=gold_pcfg_file, add_noise=add_noise,
                               noise_sigma=noise_sigma)
        elif init_strategy:
            logging.info("Initialization strategy found \"{}\". Executing strategy.".format(init_strategy))
            if init_strategy in STRATEGY_STRINGS:
                pcfg_replace_model(None, None, models, strategy=STRATEGY_STRINGS[init_strategy],
                                   ints_seqs=sample.ev_seqs
                                   , gold_pos_dict=gold_pos_dict)
            else:
                raise Exception("strategy {} not found!".format(init_strategy))
        else:
            pcfg_replace_model(None, None, models, pcfg_model)

        sample.models = models
        iter = 0

    else:
        sample = uhhmm_io.read_serialized_sample(pickle_file)
        sample.log_prob = 0
        models = sample.models
        hid_seqs = sample.hid_seqs
        max_state_check(hid_seqs, models, "reading sample from pickle file")

        pcfg_model.set_log_mode('a')
        pcfg_model.start_logging()

        inflated_num_abp = models.A[0].dist.shape[-1]
        iter = sample.iter + 1
        pcfg_model.iter = iter

    indexer = Indexer(models)

    stats = Stats()
    inf_procs = list()

    workDistributer = WorkDistributerServer(ev_seqs, working_dir)
    logging.info("GPU is %s with %d workers and batch size %d" % (gpu, num_gpu_workers, gpu_batch_size))
    logging.info("Start a new worker with python3 scripts/workers.py %s %d %d %d %d %d %d" % (
    workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port,
    maxLen + 1, int(gpu), gpu_batch_size))

    ## Initialize all the sub-processes with their input-output queues
    ## and dimensions of matrix they'll need
    if num_cpu_workers + num_gpu_workers > 0:
        inf_procs = start_local_workers_with_distributer(workDistributer, maxLen, num_cpu_workers, num_gpu_workers, gpu,
                                                         gpu_batch_size)
        signal.signal(signal.SIGINT, lambda x, y: handle_sigint(x, y, inf_procs))

    elif cluster_cmd != None:
        start_cluster_workers(workDistributer, cluster_cmd, maxLen, gpu)
    else:
        master_config_file_gpu = os.path.join(working_dir, 'masterConfig_gpu.txt')
        master_config_file_cpu = os.path.join(working_dir, 'masterConfig_cpu.txt')
        with open(master_config_file_gpu, 'w') as c:
            print(' '.join([str(x) for x in
                            [workDistributer.host, workDistributer.jobs_port, workDistributer.results_port,
                             workDistributer.models_port, maxLen + 1, 1, gpu_batch_size]]), file=c)
            print('OK', file=c)
        with open(master_config_file_cpu, 'w') as c:
            print(' '.join([str(x) for x in
                            [workDistributer.host, workDistributer.jobs_port, workDistributer.results_port,
                             workDistributer.models_port, maxLen + 1, 0, gpu_batch_size]]), file=c)
            print('OK', file=c)


    logging.info("Starting workers")
    max_loglikelihood = -np.inf
    best_init_model = None
    best_anneal_model = None
    best_anneal_likelihood = -np.inf
    prev_anneal_coeff = -np.inf
    acc_logprob = 0
    # compile the initialized the models
    compile_and_set_models(depth, workDistributer, gpu, init_depth, models, working_dir)
    ### Start doing actual sampling:
    while num_samples < max_samples:
        # this file is used to control CPU workers to not load models whenever the file is present
        # which indicates that GPU parsing is going on
        if os.path.exists(PARSING_SIGNAL_FILE):
            os.remove(PARSING_SIGNAL_FILE)

        sample.iter = iter
        pcfg_model.iter = iter

        ## If user specified an init depth (i.e. less than the final depth), increment it after 1000 iterations.
        if init_depth < depth:
            if iter > 0 and iter % 1000 == 0:
                init_depth += 1

        split_merge = False
        if split_merge_iters > 0 and iter >= burnin and (iter - burnin) % split_merge_iters == 0:
            split_merge = True

        if finite:
            if iter > 0 and np.random.random() < infinite_sample_prob:
                if finite:
                    logging.info("Randomly chose infinite sample with probability %f" % infinite_sample_prob)
                finite = False
                return_to_finite = True
            else:
                logging.info("Performing standard finite sample")

        # num_abp + 2
        inflated_num_abp = models.A[0].dist.shape[-1]

        totalK = indexer.get_state_size()

        ## Give the models to the model server. (Note: We used to pass them to workers via temp files which works when you create new
        ## workers but is harder when you need to coordinate timing of reading new models in every pass)

        t0 = time.time()

        this_log_prob, num_processed = parse(start_ind, end_ind, workDistributer, ev_seqs, hid_seqs)

        sample.log_prob += this_log_prob
        acc_logprob += sample.log_prob
        pcfg_model.log_probs = acc_logprob
        # random restarts control
        if iter < random_restarts - 1:
            logging.info("The {} random restart has a loglikelihood of {}".format(iter, sample.log_prob))
            checkpoint_function(sample)
            if sample.log_prob > max_loglikelihood:
                max_loglikelihood = sample.log_prob
                best_init_model = copy.deepcopy(models)
            pcfg_replace_model(None, None, models, pcfg_model)
            compile_and_set_models(depth, workDistributer, gpu, init_depth, models, working_dir)
            sample.log_prob = 0
            acc_logprob = 0
            pcfg_model.log_probs = 0
            iter += 1
            continue
        elif iter == random_restarts - 1:
            checkpoint_function(sample)
            if sample.log_prob > max_loglikelihood:
                max_loglikelihood = sample.log_prob
                best_init_model = copy.deepcopy(models)
            # best_init_model = unreanneal(best_init_model, next_ac_coeff=init_anneal_likelihood)
            sample.models = best_init_model
            models = best_init_model
            compile_and_set_models(depth, workDistributer, gpu, init_depth, models, working_dir)
            if validation:  # validation is after annealing
                validation_prob, _ = parse(num_ev_sents, num_ev_sents + abs(validation_length), workDistributer,
                                           ev_seqs, hid_seqs)
                pcfg_model.val_log_probs = validation_prob
            logging.info("The {} random restart has a loglikelihood of {}".format(iter, sample.log_prob))
            logging.info("The best init model has a loglikehood of {}. Will be using this for sampling.".format(
                max_loglikelihood))
            max_loglikelihood = -np.inf
            sample.log_prob = 0
            acc_logprob = 0
            pcfg_model.log_probs = 0
            iter += 1

            continue
        else:
            pass

        if num_processed < (end_ind - start_ind):
            logging.warning("Didn't receive the correct number of parses at iteration %d" % iter)

        t1 = time.time()
        # logging.info("Building counts tables took %d s" % (t1 - t0))

        if iter >= burnin and (iter - burnin) % iters == 0:
            logging.info("Performed enough batches to collect a sample -- waiting for complete pass to finish")
            ready_for_sample = True

        if end_ind >= num_ev_sents:
            if end_ind > num_ev_sents:
                logging.warning("There are more parses than input sentences!")

            logging.info("Finished complete pass through data -- calling checkpoint function")
            sample.hid_seqs = hid_seqs
            checkpoint_function(sample)
            # always sample
            if always_sample:
                ready_for_sample = True

            if ready_for_sample:
                # logging.info("Collecting sample")
                # samples.append(sample)
                report_function(sample)
                logging.info(".\n")
                num_samples += 1
                ready_for_sample = False

            next_sample = Sample()
            # next_sample.hid_seqs = hid_seqs

            next_sample.ev_seqs = ev_seqs

            prev_sample = sample
            sample = next_sample

        max_state_check(hid_seqs, models, "sampling hyperparameters")

        t0 = time.time()

        cur_anneal_iter = iter - random_restarts
        if cur_anneal_iter >= anneal_length:  # if annealing is finished
            logging.warn(
                "iteration index {} is larger than annealing length {}! Doing normal sampling!".format(iter,
                                                                                                            anneal_length))
            mh_counts = 0
            while True:
                mh_counts += 1
                old_models = copy.deepcopy(models)
                pcfg_replace_model(hid_seqs[:validation_length], ev_seqs[:validation_length], old_models, pcfg_model, sample_alpha_flag=sample_alpha_flag)
                compile_and_set_models(depth, workDistributer, gpu, init_depth, old_models, working_dir)

                if validation: # validation is after annealing
                    validation_prob, _ = parse(num_ev_sents, num_ev_sents + abs(validation_length), workDistributer,
                                               ev_seqs, hid_seqs)
                    thres = np.log10(np.random.uniform(0, 1))
                    logging.info("try {}: val likelihood: {}; prev val likelihood: {}; thres {}".format(mh_counts,
                                                                                                        validation_prob, pcfg_model.val_log_probs, thres))
                    if (pcfg_model.val_log_probs == -np.inf or validation_prob - pcfg_model.val_log_probs > thres) or validation_no_mh:
                        pcfg_model.val_log_probs = validation_prob
                        pcfg_model.mh_tries = mh_counts
                        models = old_models
                        pcfg_model.write_params()
                        break

                else:
                    models = old_models
                    pcfg_model.write_params()
                    break
            if validation:
                logging.info("the dev log prob for this iteration {} is {}, which takes {} tries.".format(iter, validation_prob, mh_counts))
            logging.info("The log prob for this iter is {}".format(acc_logprob))

        elif super_cooling and super_cooling_start_iter >= iter:
            cur_cooling_iter = iter - super_cooling_start_iter
            ac_coeff = calc_simulated_annealing(cur_cooling_iter, super_cooling_length, 1,
                                                super_cooling_target_ac)
            logging.info("inside super cooling phase with current ac coeff {}".format(ac_coeff))
            pcfg_replace_model(hid_seqs[:validation_length], ev_seqs[:validation_length], models, pcfg_model, ac_coeff=ac_coeff,
                               annealing_normalize=normalize_flag, sample_alpha_flag=sample_alpha_flag)
            compile_and_set_models(depth, workDistributer, gpu, init_depth, models, working_dir)

        else:
            ac_coeff = calc_simulated_annealing(cur_anneal_iter, anneal_length, init_anneal_likelihood,
                                                final_anneal_likelihood, anneal_likelihood_phase)
            # annealing control
            if MAP:
                next_ac_coeff = calc_simulated_annealing(cur_anneal_iter + 1, anneal_length, init_anneal_likelihood,
                                                         final_anneal_likelihood, anneal_likelihood_phase)
                if next_ac_coeff != ac_coeff:
                    logging.info("The annealing coeff will jump from {} to {}".format(ac_coeff, next_ac_coeff))
                    if acc_logprob > best_anneal_likelihood:
                        best_anneal_likelihood = acc_logprob
                        best_anneal_model = copy.deepcopy(models)
                    logging.info("The best model at {} has likelihood of {} ".format(ac_coeff, best_anneal_likelihood))
                    # real_model = unreanneal(best_anneal_model, ac_coeff, next_ac_coeff)
                    real_model = best_anneal_model
                    sample.models = real_model
                    models = real_model
                    best_anneal_likelihood = -np.inf
                else:
                    logging.info(
                        "The log prob for this iter is {} and the best anneal likelihood for this phase is {}".format(
                            prev_sample.log_prob, best_anneal_likelihood))
                    if prev_sample.log_prob > best_anneal_likelihood:
                        best_anneal_likelihood = acc_logprob
                        best_anneal_model = copy.deepcopy(models)
                    pcfg_replace_model(hid_seqs[:validation_length], ev_seqs[:validation_length], models, pcfg_model, ac_coeff=ac_coeff,
                                       annealing_normalize=normalize_flag, sample_alpha_flag=sample_alpha_flag)
                    # resample_all(models, sample, params, depth, anneal_alphas, ac_coeff, normalize_flag)
            else:
                logging.info("The log prob for this iter is {}".format(acc_logprob))
                pcfg_replace_model(hid_seqs[:validation_length], ev_seqs[:validation_length], models, pcfg_model, ac_coeff=ac_coeff,
                                   annealing_normalize=normalize_flag, sample_alpha_flag=sample_alpha_flag)
                # resample_all(models, sample, params, depth, anneal_alphas, ac_coeff, normalize_flag)
            compile_and_set_models(depth, workDistributer, gpu, init_depth, models, working_dir)

        acc_logprob = 0
        ## Update sentence indices for next batch:
        if end_ind == len(ev_seqs[:validation_length]):
            start_ind = 0
            end_ind = min(len(ev_seqs[:validation_length]), batch_size)
        else:
            start_ind = end_ind
            end_ind = start_ind + min(len(ev_seqs[:validation_length]), batch_size)
            if end_ind > num_ev_sents:
                end_ind = num_ev_sents

        sample.models = models

        iter += 1

    logging.debug("Ending sampling")
    workDistributer.stop()

    for cur_proc in range(0, num_procs):
        logging.info("Sending terminate signal to worker {} ...".format(cur_proc))
        inf_procs[cur_proc].terminate()

    for cur_proc in range(0, num_procs):
        logging.info("Waiting to join worker {} ...".format(cur_proc))
        inf_procs[cur_proc].join()
        inf_procs[cur_proc] = None

    return (samples, stats)


def max_state(hid_seqs):
    return max(a.g for b in hid_seqs if b for a in b)


def max_state_check(hid_seqs, models, location):
    ms = max_state(hid_seqs)
    logging.debug("max state in hid_seqs = " + str(ms) + ", pos dist size = " + str(
        models.pos.dist.shape[1]) + ", after " + location)
    assert ms <= models.pos.dist.shape[1] - 2, \
        "Too many states in hid_seqs: max state of hid_seqs is %s, pos dist size is %s" % (ms, models.pos.dist.shape[1])
    if (ms < models.pos.dist.shape[1] - 2):
        logging.warning("Too few states in hid_seqs: there is at least one state that is never used during inference.")


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
    model.pairCounts = np.insert(model.pairCounts, model.pairCounts.shape[-1], np.zeros(model.pairCounts.shape[0:-1]),
                                 model.pairCounts.ndim - 1)

    param_a = np.tile(model.alpha * model.beta[-2], model.dist.shape[0:-1])
    param_b = model.alpha * (1 - model.beta[0:-1].sum())

    if param_b < 0:
        param_a_str = str(param_a)
        param_b_str = str(param_b)
        logging.error("param b is < 0! inputs are alpha=%f, beta=%s, betasum=%f" % (
        model.alpha, str(model.beta), model.beta[0:-1].sum()))
        logging.info("Param a=%s and param b=%s" % (param_a_str, param_b_str))
        param_b = 0

    if param_a.min() < 1e-2 or param_b < 1e-2:
        pg = np.random.binomial(1, param_a / (param_a + param_b))
    else:
        pg = np.random.beta(param_a, param_b)

    ## Copy the last dimension into the new dimension:
    model.dist = np.insert(model.dist, model.dist.shape[-1], model.dist[..., -1], model.dist.ndim - 1)
    model.dist[..., -2] += np.log10(pg)
    model.dist[..., -1] += np.log10(1 - pg)

    if np.argwhere(np.isnan(model.dist)).size > 0:
        logging.error("Addition of column resulted in nan!")


def add_model_row_simple(model, base):
    num_outs = model.dist.shape[1]
    model.pairCounts = np.append(model.pairCounts, np.zeros((1, num_outs)), 0)
    model.dist = np.append(model.dist, np.zeros((1, num_outs)), 0)
    model.dist[-1, 0] = -np.inf
    model.dist[-1, 1:] = np.log10(sampler.sampleSimpleDirichlet(model.pairCounts[-1, 1:] + base))
    if np.argwhere(np.isnan(model.dist)).size > 0:
        logging.error("Addition of column resulted in nan!")




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
        logging.error("Initialization sequence length %d is different than input sequence length %d." % (
        len(init_seqs), len(ev_seqs)))
        raise Exception

    state_seqs = list()
    for sent_index, sent in enumerate(ev_seqs):
        hid_seq = list()
        if len(sent) != len(init_seqs[sent_index]):
            logging.error("Sentence %d has length %d in init but length %d in input." % (
            sent_index, len(init_seqs[sent_index], len(sent))))
            raise Exception

        for index, word in enumerate(sent):
            state = State.State(max_depth)
            state.f = init_seqs[sent_index][index].f
            state.j = init_seqs[sent_index][index].j
            for d in range(0, input_depth):
                state.a[d] = init_seqs[sent_index][index].a[d]
                state.b[d] = init_seqs[sent_index][index].b[d]
            state.g = init_seqs[sent_index][index].g

            hid_seq.append(state)
        state_seqs.append(hid_seq)
        pcfg_increment_counts(hid_seq, sent, models)
        # increment_counts(hid_seq, sent, models)
    return state_seqs


# Randomly initialize all the values for the hidden variables in the
# sequence. Obeys constraints (e.g., when f=1,j=1 a=a_{t-1}) but otherwise
# samples randomly.
RANDOM_INIT = 0
RB_INIT = 1
LB_INIT = 2


def decrement_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    pcfg_increment_counts(hid_seqs[start_ind:end_ind], sents[start_ind:end_ind], models, -1)


def increment_sentence_counts(hid_seqs, sents, models, start_ind, end_ind):
    pcfg_increment_counts(hid_seqs[start_ind:end_ind], sents[start_ind:end_ind], models, 1)

# normalize a logged matrix
def normalize(matrix):
    # assert len(matrix.shape) == 2, "shape of the normalizing matrix {} is not 2!".format(str(matrix.shape))
    # print(matrix, 'first')
    matrix = np.nan_to_num(matrix)
    # matrix[ matrix == -np.inf] = 0
    matrix = 10 ** matrix
    sums = np.sum(matrix, axis=-1, keepdims=True)
    sums = np.repeat(sums, matrix.shape[-1], axis=-1)
    matrix /= sums
    matrix = np.log10(matrix)
    matrix = np.nan_to_num(matrix)
    # print(matrix, 'last')
    # assert np.sum(matrix) == np.cumprod(matrix.shape)[-1], "{}, {}".format(np.sum(matrix), np.cumprod(matrix.shape)[-1])
    return matrix

