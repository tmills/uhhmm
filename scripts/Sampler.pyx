#!/usr/bin/env python3

import ihmm
import logging
import time
import numpy as np
import sys, os, linecache
from multiprocessing import Process,Queue,JoinableQueue
#import pyximport; pyximport.install()
import log_math as lm

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

class Sampler:
    def __init__(self, seed=-1):
        self.ff_time = 0
        self.bs_time = 0
        if seed > 0:
            np.random.seed(seed)

    def sample(self, sent, sent_index):
        
        try:
            logging.debug("Starting forward pass of sentence %d with length %d" % (sent_index, len(sent)))
            log_prob = self.forward_pass(sent, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % sent_index)
            sent_sample = self.reverse_sample(sent, sent_index)
            logging.debug("Finished parsing sentence %d" % sent_index)
        except Exception as e:
            printException()
            raise e
                
        return (sent_sample, log_prob)
