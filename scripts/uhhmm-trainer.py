#!/usr/bin/env python3.4

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import calcV
import uhhmm_io as io
import configparser
import uhhmm
import numpy as np
import pdb
import os
import pickle
import logging
import signal
from random import randint

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <Config file|Resume directory>\n")
        sys.exit(-1)

    path = argv[0]
    if not os.path.exists(path):
        sys.stderr.write("Input file/dir does not exist!")
        sys.exit(-1)
    
    config = configparser.ConfigParser()
    pickle_file = None
    
    if os.path.isdir(path):
        ## Resume mode
        config.read(path + "/config.ini")
        pickle_file = path + "/sample.obj"
        out_dir = config.get('io', 'output_dir')
    else:
        config.read(argv[0])
        out_dir = config.get('io', 'output_dir')
        if not os.path.exists(out_dir):
            sys.stderr.write("Creating non-existent output directory.\n")
            os.makedirs(out_dir)

        with open(out_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
    
    input_file = config.get('io', 'input_file')
    working_dir = config.get('io', 'working_dir', fallback=out_dir)

    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = io.read_input_file(input_file)
    num_types = max(map(max,word_seq)) + 1

    params = read_params(config)
    
    params['h'] = init_emission_base(num_types)
    
    ## Store tag sequences of gold tagged sentences
    gold_seq = dict()
    while len(gold_seq) < int(params.get('num_gold_sents', 0)) and len(gold_seq) < len(word_seq):
      rand = randint(0,len(word_seq)-1)
      if rand not in gold_seq.keys():
        gold_seq[rand]=pos_seq[rand]
    
    (samples, stats) = uhhmm.sample_beam(word_seq, params, lambda x: io.write_output(x, None, config, pos_seq), lambda x: io.checkpoint(x,config), working_dir, pickle_file, gold_seq)
    
    if len(samples) > 0:
        io.write_output(samples[-1], stats, config, pos_seq)

def read_params(config):
    params = {}
    for (key, val) in config.items('params'):
        params[key] = val
    
    return params

def init_emission_base(size):
    ## Uniform distribution:
    H = np.zeros((1,size)) + 0.01
    return H


    
if __name__ == "__main__":
    main(sys.argv[1:])
