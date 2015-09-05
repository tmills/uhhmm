#!/usr/bin/env python3

import ihmm
import logging
import time
import numpy as np
import sys
from multiprocessing import Process,Queue,JoinableQueue
import pyximport; pyximport.install()
import log_math as lm

# This class does the actual sampling. It is a Python process rather than a Thread
# because python threads do not work well due to the global interpreter lock (GIL), 
# which only allows one thread at a time to access the interpreter. Making it a process
# is requires more indirect communicatino using shared input/output queues between 
# different sampler instances

class Sampler(Process):
    def __init__(self, in_q, out_q, models, totalK, maxLen, tid, out_freq=100):
        Process.__init__(self)
        self.in_q = in_q
        self.out_q = out_q
        self.models = models
        self.K = totalK
        self.dyn_prog = []
        self.tid = tid
        self.out_freq = out_freq
    
    def set_data(self, sent):
        self.sent = sent

    def run(self):
        self.dyn_prog[:,:] = -np.inf
        #logging.debug("Starting forward pass in thread %s", self.tid)

        while True:
            task = self.in_q.get()
            if task == None:
                self.in_q.task_done()
                break
            
            (sent_index, sent) = task
            t0 = time.time()
            self.dyn_prog[:,:] = -np.inf
            (self.dyn_prog, log_prob) = self.forward_pass(self.dyn_prog, sent, self.models, self.K, sent_index)
            sent_sample = self.reverse_sample(self.dyn_prog, sent, self.models, self.K, sent_index)
            if sent_index % self.out_freq == 0:
                logging.info("Processed sentence {0}".format(sent_index))

            t1 = time.time()
            self.in_q.task_done()
            self.out_q.put((sent_index, sent_sample,log_prob))
            
            if log_prob > 0:
                logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))
            
            #logging.debug("Thread %d required %d s to process sentence.", self.tid, (t1-t0))

    def get_sample(self):
        return self.sent_sample
    
