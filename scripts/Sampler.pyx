#!/usr/bin/env python3

import logging
import numpy as np
from uhhmm_io import printException
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

class Sampler:        
    def __init__(self, seed=-1):
        self.ff_time = 0
        self.bs_time = 0
        if seed > 0:
            np.random.seed(seed)

    def sample(self, gpu_pi, list sents, int sent_index):
        
        #logging.info("Converting transition matrix to gpu")
        #gpu_pi = gpuarray.to_gpu(pi.astype(np.float32).toarray())
        try:
            logging.debug("Starting forward pass of sentence %d with batch size %d" % (sent_index, len(sents)))
            log_prob, forward = self.forward_pass(gpu_pi, sents, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % (sent_index) )
            sent_sample = self.reverse_sample(forward, gpu_pi, sents, sent_index)
            logging.debug("Finished parsing sentence %d" % (sent_index) )
        except Exception as e:
            printException()
            raise e
                
        return (sent_sample, log_prob)
