#!/usr/bin/env python3

import logging
import numpy as np
from uhhmm_io import printException
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

cdef class Sampler:
    def __cinit__(self):
        self.ff_time = 0
        self.bs_time = 0
        
    def __init__(self, seed=-1):
        if seed > 0:
            np.random.seed(seed)

    cpdef sample(self, pi, list sent, int sent_index):
        
        #logging.info("Converting transition matrix to gpu")
        gpu_pi = gpuarray.to_gpu(pi.astype(np.float32).toarray())
        try:
            logging.debug("Starting forward pass of sentence %d with length %d" % (sent_index, len(sent)))
            log_prob, forward = self.forward_pass(gpu_pi, sent, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % (sent_index) )
            sent_sample = self.reverse_sample(forward, gpu_pi, sent, sent_index)
            logging.debug("Finished parsing sentence %d" % (sent_index) )
        except Exception as e:
            printException()
            raise e
                
        return (sent_sample, log_prob)
