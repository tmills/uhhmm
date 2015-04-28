#!/usr/bin/env python3.4

import sys
import ConfigParser
import ihmm
import logging
import numpy as np

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <Config file>\n")
        sys.exit(-1)

    config = ConfigParser.ConfigParser()
    config.read(argv[0])
    
    input_file = config.get('io', 'input_file')
        
    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = read_input_file(input_file)
    num_types = max(map(max,word_seq)) + 1
    
    params = read_params(config)
    
    params['h'] = init_emission_base(num_types)
    
    (state_seq, stats) = ihmm.sample_beam(word_seq, params)
    
    write_output(state_seq, stats, config)        

def read_params(config):
    params = {}
    for (key, val) in config.items('params'):
        #logging.debug("assigning params key val pair (%s, %s)", key, val)
        params[key] = val
    
    return params

def init_emission_base(size):
    ## Uniform distribution:
    H = np.zeros((1,size)) + 0.01
    return H

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

def write_output(state_seq, stats, config):
    output_dir = config.get('io', 'output_dir')
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

