#!/usr/bin/env python3.4

## To compute V-measure (as in vangael et al.)
## V = [(1 + Beta) * h * c ] / [(Beta*h) + c]
## where:
## h = 1 - H(C|K) / H(C)
## c = 1 - H(K/C) / H(K)
## H(X|Y) is conditional entropy:
## sum (x,y) p(x,y) * log [ p(x) / p(x,y) ]
## and H(X) is entropy:
## sum (x) p(x) * log p(x)
## so we need to read in taggings from two different systems 
## (usually automatically tagged vs. gold and calculate three
## probability distributions;
## p(x,y), p(x), and p(y)
## and everything else is derived from those.
## Beta is a parameter which balances the extent to which the 
## measure penalizes or rewards large clusters.
## Beta = 1 is V measure
## general term is V-beta.

## Expected input format is one sentence per line, with
## a sentence represented as space-separated POS tags.
## They should be mapped to integer indices, e.g.:
## 15 2 5 2 3
## is a five-word sentence.
## The system and gold input must be aligned or the output will
## be garbage.

import numpy as np
import ihmm_io as io
import itertools
import sys
import pdb

def main(args):
    if sys.version_info[0] != 3:
        print("This script requires Python 3")
        exit()

    gold_pos = io.read_input_file(args[0])[1]
    sys_pos = io.read_input_file(args[1])[1]
    
    p_x = get_distribution(gold_pos)
    p_y = get_distribution(sys_pos)
    p_xy = get_joint_distribution(gold_pos, sys_pos)
    
    v = get_v(p_x, p_y, p_xy)
    
    print("Value of V is %f\n" % v)
    
def get_v(p_x, p_y, p_xy, beta=1):
    p_x_int = p_x
    p_x_int[p_x==0] = 1
    p_y_int = p_y
    p_y_int[p_y==0] = 1
        
    H_x = sum(p_x * np.log(p_x_int))
    H_y = sum(p_y * np.log(p_y_int))
    
    ## This is a bit tricky: We are dividing a 1d array by a 2d array (matrix)
    ## and x and y have to divide in different directions because the matrix is
    ## x/y in each dimension. Further, you can't take the transpose of a 1d array
    ## and have it change the outcome. So i had to tranpose the matrix, then do the
    ## divide, then transpose the outcome.
    
    ## This is wonky but it is necessary to avoid dividing by zero and taking the log
    ## of zero. Just make a copy of p_xy where all the 0s are 1s for the internal
    ## calculation. Those values will be zeroed out by the outer multiplication which
    ## uses the original matrix with all the zeros.
    p_xy_int = p_xy
    p_xy_int[p_xy == 0] = 1
    
    H_xy = (p_xy * np.log((p_x / p_xy_int.transpose()).transpose())).sum()    
    H_yx = (p_xy * np.log(p_y / p_xy_int)).sum()
    
    h = 1 - (H_xy / H_x)
    c = 1 - (H_yx / H_y)
    
    V = ((1 + beta) * h * c ) / ((beta*h) + c)
    
    return V

def get_distribution(seq_list):
    flat_seq = list(itertools.chain.from_iterable(seq_list))
    max_val = max(flat_seq)
    counts = np.bincount(flat_seq)
    
    return counts / counts.sum()
    
def get_joint_distribution(seqs1, seqs2):
    seq1_flat = list(itertools.chain.from_iterable(seqs1))
    seq2_flat = list(itertools.chain.from_iterable(seqs2))
    assert len(seq1_flat) == len(seq2_flat)

    ## Add 1 to size variable because we assume 0-indexing of input.
    counts = np.zeros((max(seq1_flat)+1, max(seq2_flat)+1))
    
    for i,val in enumerate(seq1_flat):
        counts[seq1_flat[i]][seq2_flat[i]] += 1

    return counts / counts.sum()

if __name__ == "__main__":
    main(sys.argv[1:])
