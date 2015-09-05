#!/usr/bin/env python3.4

import calcV
import numpy as np
import pickle

def read_input_file(filename):
    pos_seqs = list()
    token_seqs = list()
    f = open(filename, 'r')
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

def read_serialized_sample(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')
    return pickle.load(pickle_file)

def write_output(sample, stats, config, gold_pos=None):
#    last_sample = samples[-1]
    models = sample.models
    
    output_dir = config.get('io', 'output_dir')
    dict_file = config.get('io', 'dict_file')
    word_dict = dict()
    if dict_file != None:
        f = open(dict_file, 'r')
        for line in f:
            #pdb.set_trace()
            (word, index) = line.rstrip().split(" ")
            word_dict[int(index)] = word
    
    if gold_pos != None:
        sys_pos = extract_pos(sample)
        v = calcV.get_v(gold_pos, sys_pos)
        f = open(output_dir + "/v_%d.txt" % sample.iter, 'w')
        f.write('%f\n' % v)
        f.close()

    f = open(output_dir + "/logprobs.txt", 'a')
    f.write('%d\t%f\n' % (sample.iter,sample.log_prob) )
    f.close()
    
    write_model(models.lex.dist, output_dir + "/p_lex_given_pos%d.txt" % sample.iter, word_dict)
    write_model(models.pos.dist, output_dir + "/p_pos_given_b_%d.txt" % sample.iter, condPrefix="AWA", outcomePrefix="POS")
    write_model(models.cont.dist, output_dir + "/p_awa_given_b+g%d.txt" % sample.iter,
    condPrefix="BG", outcomePrefix="AWA")
    write_model(models.start.dist, output_dir + "/p_awa_given_a%d.txt" % sample.iter,
    condPrefix="AA", outcomePrefix="AWA")
    write_model(models.act.dist, output_dir + "/p_act_given_a%d.txt" % sample.iter,
    condPrefix="ACT", outcomePrefix="ACT")
    write_model(models.root.dist, output_dir + "/p_act_given_g%d.txt" %sample.iter,
    condPrefix="POS", outcomePrefix="ACT")
    write_model(models.fork.dist, output_dir + "/p_fork_given_b+g%d.txt" % sample.iter,
    condPrefix="BG", outcomePrefix="F")
    write_model(models.reduce.dist, output_dir + "/p_join_given_a+f1_%d.txt" % sample.iter,
    condPrefix="A+", outcomePrefix="J")
    
    write_last_sample(sample, output_dir + "/last_sample%d.txt" % sample.iter)
    
    out_file = open(output_dir + "/sample.obj", 'wb')
    pickle.dump(sample, out_file)
    out_file.close()
    
def write_model(dist, out_file, word_dict=None, condPrefix="", outcomePrefix=""):
    f = open(out_file, 'w')
    
    for ind,val in np.ndenumerate(dist):
        lhs = ind[0:-1]
        rhs = ind[-1]
        if rhs == 0:
            continue

        if word_dict == None:
            f.write("P( %s%d | %s%s ) = %f \n" % (outcomePrefix, rhs, condPrefix, str(lhs), 10**val))
        else:
            f.write("P( %s | %s ) = %f \n" % (word_dict[rhs], str(lhs), 10**val))
                
    f.close()

def write_last_sample(sample, out_file):
    f = open(out_file, 'w')
    #pdb.set_trace()
    for sent_state in sample.hid_seqs:
        state_str = str(list(map(lambda x: x.str(), sent_state)))
        f.write(state_str)
        f.write('\n')

def extract_pos(sample):
    pos_seqs = list()
    for sent_state in sample.hid_seqs:
        pos_seqs.append(list(map(lambda x: x.g, sent_state)))
    
    return pos_seqs