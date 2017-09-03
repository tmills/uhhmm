from EMSampler import EMSampler
import logging
import Indexer
import DistributedModelCompiler
import uhhmm_io as io
import os
import nltk.grammar
from collections import defaultdict
from models import Model, Models
from pcfg_translator import *
from WorkDistributerServer import WorkDistributerServer
from workers import start_local_workers_with_distributer, start_cluster_workers
import signal
import argparse
import pickle
import sys
from global_params import PARSING_SIGNAL_FILE
from ViterbiParser import ViterbiParser

def em(last_sample_directory, param_iter, line_intstok_file, dict_file, abp_domain_size=15, depth=2, em_cycles=100):
    '''
    This is the main function to parse some data given some model. The model is from some row in the pcfg nonterm and
    term param files. The param iter is the row or iteration of the files. The line_tok_file is the need-to-parse
    corpus converted into integers using the original grammar's dict. This function does not do preprocessing.
    :param last_sample_directory: the directory of some trained grammar: path
    :param param_iter: the iteration number for a particular grammar
    :param line_intstok_file: the ints file of the corpus to be parsed
    :param dict_file: the dict file for the line ints file
    :param abp_domain_size: the size of the abp domain in the grammar
    :return: write out a file in the format of pcfg bracketed trees named last_sample0.linetrees for evaluation purposes
    '''
    target_parse_file = os.path.join(last_sample_directory, 'last_sample0.linetrees')
    terms_param_file = os.path.join(last_sample_directory, 'pcfg_terms.txt')
    nonterms_param_file = os.path.join(last_sample_directory, 'pcfg_nonterms.txt')
    _, word_seq = io.read_input_file(line_intstok_file)
    # read in the params and dict
    with open(terms_param_file) as terms_fh:
        terms_header = terms_fh.readline().strip().split('\t')[1:]
        for line in terms_fh:
            if line.startswith(str(param_iter)):
                terms_param = line.strip().split()[1:]
                break
        else:
            raise Exception('no iter number found in param files.')
    with open(nonterms_param_file) as nonterms_fh:
        nonterms_header = nonterms_fh.readline().strip().split('\t')[1:]
        for line in nonterms_fh:
            if line.startswith(str(param_iter)):
                nonterms_param = line.strip().split()[1:]
                break
        else:
            raise Exception('no iter number found in param files.')
    with open(dict_file, 'r', encoding='utf-8') as dict_fh:
        word_dict = {}
        for line in dict_fh:
            word, index = line.rstrip().split(" ")
            word_dict[word] = index
    # construct a pcfg
    pcfg = defaultdict(dict)
    nonterms = [nltk.grammar.Nonterminal(str(x)) for x in range(abp_domain_size + 1)]
    terms_probs = zip(terms_header, terms_param)
    nonterms_probs = zip(nonterms_header, nonterms_param)
    for rule_prob_pair in nonterms_probs:
        rule, prob = rule_prob_pair
        left, right = rule.split('->')
        right1, right2 = right.replace('(', '').replace(')', '').split(', ')
        pcfg[nonterms[int(left)]][(nonterms[int(right1)], nonterms[int(right2)])] = float(prob)

    for lex_prob_pair in terms_probs:
        lex, prob = lex_prob_pair
        pos, lex = lex.split('->')
        if lex == '-ROOT-':
            pcfg[nonterms[int(pos)]][(lex,)] = float(prob)
            continue
        pcfg[nonterms[int(pos)]][(word_dict[lex],)] = float(prob)

    uhhmm_model = Models()
    inflated_num_abp = abp_domain_size + 2
    num_types = max(map(max, word_seq)) + 1 # max index of words
    max_len = max(map(len, word_seq))
    # params is not needed, corpus_shape is not needed (relic from infinite model)
    uhhmm_model = initialize_models(uhhmm_model, num_types, None, (1, 1), depth, inflated_num_abp)
    # only the pcfg dict model is provided, nothing else is needed


    # default settings:
    gpu = True
    num_gpu_workers = 1
    gpu_batch_size = 1
    num_cpu_workers = 5
    work_distributer = WorkDistributerServer(word_seq, last_sample_directory)
    logging.info("GPU is %s with %d workers and batch size %d" % (gpu, num_gpu_workers, gpu_batch_size))
    logging.info("Start a new worker with python3 scripts/workers.py %s %d %d %d %d %d %d" % (
        work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port,
    max_len + 1, int(gpu), gpu_batch_size))

    ## Initialize all the sub-processes with their input-output queues
    ## only support local workers

    if num_cpu_workers + num_gpu_workers > 0:
        inf_procs = start_local_workers_with_distributer(work_distributer, max_len, num_cpu_workers, num_gpu_workers, gpu,
                                                         gpu_batch_size)
        signal.signal(signal.SIGINT, lambda x, y: handle_sigint(x, y, inf_procs))

    # the main em cycles
    for i in range(em_cycles):
        pcfg_replace_model(None, None, uhhmm_model, None, sampled_pcfg = pcfg)

        model_name = compile_and_set_models(depth, work_distributer, gpu, depth, uhhmm_model, last_sample_directory, viterbi=True)
        model = pickle.load(open(model_name, 'rb'))
        em_sampler = EMSampler()
        P = em_sampler.batch_sample(0, len(word_seq), model, word_seq)
        pcfg = em_sampler.collect_counts(P, pcfg)

    word_dict = {int(word_dict[x]):x for x in word_dict}
    io.write_last_sample((parses, word_seq), target_parse_file, word_dict)

    work_distributer.stop()

    for cur_proc in range(0, len(inf_procs)):
        logging.info("Sending terminate signal to worker {} ...".format(cur_proc))
        inf_procs[cur_proc].terminate()

    for cur_proc in range(0, len(inf_procs)):
        logging.info("Waiting to join worker {} ...".format(cur_proc))
        inf_procs[cur_proc].join()
        inf_procs[cur_proc] = None

    return parses

if __name__ == '__main__':
    # last_sample_directory, param_iter, line_intstok_file, dict_file, abp_domain_size = 15, depth = 2
    parser = argparse.ArgumentParser(description='UHHMM GPU parser')
    parser.add_argument('-dir', type=str, help='the directory where the last sample files are')
    parser.add_argument('-iter', type=int, help='the iter number for the model for use')
    parser.add_argument('-parse-file', type=str, help='the path and name of the ints file wanting to be parsed')
    parser.add_argument('-dict-file', type=str, help='the path and name of the dict file for the ints file and original'
                                                     'training set')
    parser.add_argument('-abp', type=int, help='the number of abp categories in the grammar')
    parser.add_argument('-depth', type=int, help='the number of depths in the grammar')
    args = parser.parse_args()
    last_sample_directory, param_iter, line_intstok_file, dict_file, abp_domain_size , depth = args.dir, args.iter, \
    args.parse_file, args.dict_file, args.abp, args.depth

    parses = em(last_sample_directory, param_iter, line_intstok_file, dict_file, abp_domain_size , depth)
    print('parsed {} '.format(len(parses)))
    print('the saved parse trees are at {}'.format(last_sample_directory+'/last_sample0.linetrees'))