#!/usr/bin/env python3.4

import calcV
import logging
import numpy as np
import os.path
import pickle
import shutil
import sys, linecache
import State

## This method reads a "last_sample*.txt" file which is space-delimited, and
## each token is formatted as f/j::ACT/AWA;^d:POS;token where ^d indicates that
## the ACT/AWA section is a semi-colon delimited list of length d. Each token
## only indicates its own depth, not the max depth.
def read_input_states(filename, depth):
    states = list()

    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        hid_seqs = []
        for str_state in line.split():
            #print(str_state)
            state = State.State(depth)
            fj, syn = str_state.split('::')
            f,j = fj.split('/')
            syn_out = syn.split(':')
            ab = syn_out[0]
            tok = ':'.join(syn_out[1:])
            ab_lists = ab.replace('ACT','').replace('AWA','').split(';')
            pos = tok.split(';')[0]
            state.f = 0 if f == '-' else 1
            state.j = 0 if j == '-' else 1
            for d in range(len(ab_lists)):
                ab_vals = ab_lists[d].split('/')
                state.a[d] = int(ab_vals[0])
                state.b[d] = int(ab_vals[1])
            state.g = int(pos.replace('POS',''))
            hid_seqs.append(state)
        states.append(hid_seqs)
    return states

## This method reads a "tagwords" file which is space-delimited, and each
## token is formatted as POS/token.
def read_input_file(filename):
    pos_seqs = list()
    token_seqs = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        pos_seq = list()
        token_seq = list()
        for token in line.split():
            if "/" in token:
                (pos, token) = token.split("/")
            else:
                pos = 0

            pos_seq.append(int(pos))
            token_seq.append(int(token))

        pos_seqs.append(pos_seq)
        token_seqs.append(token_seq)

    return (pos_seqs, token_seqs)

def read_sample_file(filename):
    pos_seqs = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        pos_seq = list()
        line = line.replace('[', '').replace(']', '').replace("'", "")
        for token in line.split(', '):
            (junk, pos) = token.split(':')
            pos_seq.append(int(pos))

        pos_seqs.append(pos_seq)

    f.close()
    return pos_seqs

def read_serialized_sample(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')
    return pickle.load(pickle_file)

def write_serialized_models(model_list, pickle_file):
    pickle.dump(model_list, pickle_file)

def read_serialized_models(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')
    return pickle.load(pickle_file)

def write_output(sample, stats, config, gold_pos=None):
#    last_sample = samples[-1]
    models = sample.models
    depth = len(models.F)

    #print("Default encoding is %s" % sys.getdefaultencoding() )

    output_dir = config.get('io', 'output_dir')
    dict_file = config.get('io', 'dict_file')
    word_dict = dict()
    if dict_file != None:
        f = open(dict_file, 'r', encoding='utf-8')
        for line in f:
            #pdb.set_trace()
            (word, index) = line.rstrip().split(" ")
            word_dict[int(index)] = word

    if gold_pos != None:
        sys_pos = extract_pos(sample)
        v = calcV.get_v(gold_pos, sys_pos)
        f = open(output_dir + "/v_measure.txt", 'a', encoding='utf-8')
        f.write('%d\t%f\n' % (sample.iter,v))
        f.close()

        vi = calcV.get_vi(gold_pos, sys_pos)
        f = open(output_dir + "/vi.txt", 'a', encoding='utf-8')
        f.write('%d\t%f\n' % (sample.iter,vi))
        f.close()

    f = open(output_dir + "/beta.txt", 'a', encoding='utf-8')
    f.write('%d\t%s\n' % (sample.iter, np.array_str(models.pos.beta)))
    f.close()

    #write_lex_model(models.lex.dist, output_dir + "/p_lex_given_pos%d.txt" % sample.iter, word_dict)
    write_model(models.lex.dist, output_dir + "/p_lex_given_pos%d.txt" % sample.iter, word_dict)
    write_model(models.pos.dist, output_dir + "/p_pos_%d.txt" % sample.iter, condPrefix="B", outcomePrefix="POS")

    for d in range(0, depth):
        write_model(models.B_J0[d].dist, output_dir + "/p_awa_j0_%d.txt" % sample.iter, condPrefix="BG", outcomePrefix="AWA", depth=d)
        write_model(models.B_J1[d].dist, output_dir + "/p_awa_j1_%d.txt" % sample.iter, condPrefix="AA", outcomePrefix="AWA", depth=d)
        # write_model(models.exp[d].dist, output_dir + "/p_awa_exp_%d.txt" % sample.iter, condPrefix="GA", outcomePrefix="AWA", depth=d)
        # write_model(models.next[d].dist, output_dir + "/p_awa_next_%d.txt" % sample.iter, condPrefix="BA", outcomePrefix="AWA", depth=d)
        write_model(models.A[d].dist, output_dir + "/p_act_%d.txt" % sample.iter, condPrefix="AB", outcomePrefix="ACT", depth=d)
        # write_model(models.root[d].dist, output_dir + "/p_act_root_%d.txt" %sample.iter, condPrefix="BG", outcomePrefix="ACT", depth=d)
        write_model(models.F[d].dist, output_dir + "/p_fork_%d.txt" % sample.iter, condPrefix="BG", outcomePrefix="F", depth=d)
        write_model(models.J[d].dist, output_dir + "/p_j_%d.txt" % sample.iter, condPrefix="AB", outcomePrefix="J", depth=d)
        # write_model(models.trans[d].dist, output_dir + "/p_j_trans_%d.txt" % sample.iter, condPrefix="BG", outcomePrefix="J", depth=d)

    write_last_sample(sample, output_dir + "/last_sample%d.txt" % sample.iter, word_dict)

def checkpoint(sample, config):
    output_dir = config.get('io', 'output_dir')

    if os.path.isfile(output_dir + "/sample.obj"):
        logging.info("Saving previous checkpoint")
        shutil.copy(output_dir +"/sample.obj", output_dir + "/sample.obj.last")

    logging.info("Creating new checkpoint")
    out_file = open(output_dir + "/sample.obj", 'wb')
    pickle.dump(sample, out_file)
    out_file.close()

    f = open(output_dir + "/logprobs.txt", 'a', encoding='utf-8')
    f.write('%d\t%f\n' % (sample.iter,sample.log_prob) )
    f.close()


def write_model(dist, out_file, word_dict=None, condPrefix="", outcomePrefix="", depth=-1):
    f = open(out_file, 'a' if depth > 0 else 'w', encoding='utf-8')
    out_dim = dist.shape[-1]

    for ind,val in np.ndenumerate(dist):
        lhs = ind[0:-1]
        rhs = ind[-1]
        unlog_val = 10**val

        if (out_dim > 2 and rhs == 0) or unlog_val < 0.000001:
            continue

        if word_dict == None:
            f.write("P( %s%d | %s%s, %d ) = %f \n" % (outcomePrefix, rhs, condPrefix, str(lhs), depth, unlog_val))
        else:
            f.write("P( %s | %s, %d ) = %f \n" % (word_dict[rhs], str(lhs), depth, unlog_val))

    f.close()

def write_lex_model(dist, out_file, word_dict=None):
    f = open(out_file, 'w', encoding='utf-8')
    out_dim = dist.shape[-1]
    for ind,val in np.ndenumerate(dist):
        lhs = ind[0:-1]
        rhs = ind[-1]
        unlog_val = 10**val

        if (out_dim > 2 and rhs == 0) or unlog_val < 0.000001:
            continue

        if word_dict == None:
            f.write("X %s : %s = %f \n" % (str(lhs), str(rhs), 10**val))
        else:
            f.write("X %s : %s = %f \n" % (str(lhs), word_dict[rhs], 10**val))

## Sample output format -- each time step (token) is represented as the following:
## F/J::Active/Awaited:Pos   (see str() method in the State() class)
## Here we add the word to the end separated by a semi-colon.
## One sentence per line, with tokens separated by spaces.
def write_last_sample(sample, out_file, word_dict):
    f = open(out_file, 'w', encoding='utf-8')
    #pdb.set_trace()
    for sent_num,sent_state in enumerate(sample.hid_seqs):
        state_str = ""
        for token_num,token_state in enumerate(sent_state):
            token_str = word_dict[ sample.ev_seqs[sent_num][token_num] ]
            state_str += token_state.str() + ';' + token_str + ' '
        f.write(state_str.rstrip())
        f.write('\n')

def extract_pos(sample):
    pos_seqs = list()
    for sent_state in sample.hid_seqs:
        pos_seqs.append(list(map(lambda x: x.g, sent_state)))

    return pos_seqs

def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

class ParsingError(Exception):
    def __init__(self, cause):
        self.cause = cause

    def __str__(self):
        return self.cause
