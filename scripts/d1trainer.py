#!/usr/bin/env python3.4

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import calcV
import ihmm_io as io
import configparser
import ihmm
import numpy as np
import pdb
import os
import pickle
import logging
import signal

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
            sys.stderr.write("Creating non-existent output directory.")
            os.makedirs(out_dir)

        with open(out_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
    
    input_file = config.get('io', 'input_file')
    working_dir = config.get('io', 'working_dir', fallback=out_dir)

    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = io.read_input_file(input_file)
    num_types = max(map(max,word_seq)) + 1
    
    params = read_params(config)
    
#    if int(params.get('depth', 1)) > 1:
#        logging.warning("This code has only been tested with depth 1 configuration.")
    
    params['h'] = init_emission_base(num_types)
    
    (samples, stats) = ihmm.sample_beam(word_seq, params, lambda x: io.write_output(x, None, config, pos_seq), lambda x: io.checkpoint(x,config), working_dir, pickle_file)
    
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

