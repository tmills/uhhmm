#!/usr/bin/env python3

import ihmm
import logging
import time
import numpy as np
import sys
from multiprocessing import Process,Queue,JoinableQueue
import pyximport; pyximport.install()
import log_math as lm

def getVariableMaxes(models):
    a_max = models.act.dist.shape[-1]
    b_max = models.start.dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    return (a_max, b_max, g_max)

def get_state_size(models):
    maxes = getVariableMaxes(models)
    (a_max,b_max,g_max) = maxes
    return 2 * 2 * a_max * b_max * g_max            

class ParsingError(Exception):
    def __init__(self, cause):
        self.cause = cause
        
    def __str__(self):
        return self.cause

class Sampler:
    def __init__(self, seed=-1):
        if seed > 0:
            np.random.seed(seed)

    def sample(self, sent, sent_index):
        
        try:
            logging.debug("Starting forward pass of sentence %d with length %d" % (sent_index, len(sent)))
            (dyn_prog, log_prob) = self.forward_pass(self.dyn_prog, sent, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % sent_index)
            sent_sample = self.reverse_sample(self.dyn_prog, sent, sent_index)
        except ParsingError as e:
            raise e
                
        return (sent_sample, log_prob)
    
