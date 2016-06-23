#!/usr/bin/env python3

import logging
import numpy as np
from uhhmm_io import printException


cdef class Sampler:
    def __cinit__(self):
        self.ff_time = 0
        self.bs_time = 0
        
    def __init__(self, seed=-1):
        if seed > 0:
            np.random.seed(seed)

    cpdef sample(self, pi, list sent, int sent_index):
        
        try:
            logging.debug("Starting forward pass of sentence %d with length %d" % (sent_index, len(sent)))
            log_prob = self.forward_pass(pi, sent, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % (sent_index) )
            sent_sample = self.reverse_sample(pi, sent, sent_index)
            logging.debug("Finished parsing sentence %d" % (sent_index) )
        except Exception as e:
            printException()
            raise e
                
        return (sent_sample, log_prob)
