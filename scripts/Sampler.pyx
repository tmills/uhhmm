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
    a_max = models.act[0].dist.shape[-1]
    b_max = models.start[0].dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    return (a_max, b_max, g_max)

def get_state_size(models):
    maxes = getVariableMaxes(models)
    (a_max,b_max,g_max) = maxes
    depth = len(models.fork)
    return ((2 * 2 * a_max * b_max) ** depth) * g_max            

class Sampler:
    def sample(self, sent, sent_index):

        (dyn_prog, log_prob) = self.forward_pass(self.dyn_prog, sent, sent_index)
        sent_sample = self.reverse_sample(self.dyn_prog, sent, sent_index)
        
        return (sent_sample, log_prob)
    
