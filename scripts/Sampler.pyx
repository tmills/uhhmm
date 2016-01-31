#!/usr/bin/env python3

import ihmm
import logging
import time
import numpy as np
import sys, os, linecache
from multiprocessing import Process,Queue,JoinableQueue
import pyximport; pyximport.install()
import log_math as lm

def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


def getVariableMaxes(models):
    a_max = models.act[0].dist.shape[-1]
    b_max = models.start[0].dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    return (a_max, b_max, g_max)

def get_state_size(models):
    maxes = getVariableMaxes(models)
    (a_max,b_max,g_max) = maxes
    depth = len(models.fork)
    return 4*depth * (a_max * b_max)**depth * g_max

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
        except Exception as e:
            printException()
            raise e
                
        return (sent_sample, log_prob)