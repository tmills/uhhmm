#!/usr/bin/env python3

import zmq
from PyzmqMessage import *
from multiprocessing import Process
import logging
import time


class PyzmqSampler(Process):
    def __init__(self, models, host, jobs_port, results_port, totalK, maxLen, tid, out_freq=100):
        Process.__init__(self)
        logging.debug("Sampler %d started" % tid)
        self.models = models
        self.host = host
        self.jobs_port = jobs_port
        self.results_port = results_port
        self.K = totalK
        self.maxLen = maxLen
        self.tid = tid
        self.out_freq = out_freq
        
    def run(self):
        logging.debug("Starting forward pass in thread %s", self.tid)

        context = zmq.Context()

        #  Socket to talk to server
        logging.debug("Connecting to work distribution server…")
        jobs_socket = context.socket(zmq.PULL)        
        jobs_socket.connect("tcp://%s:%s" % (self.host, self.jobs_port))

        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%s" % (self.host, self.results_port))
        
        logging.debug("Worker connected to both endpoints")
        
        while True: 
            logging.debug("Worker waiting for job")
            job = jobs_socket.recv_pyobj();
            sent_index = job.index
            sent = job.ev_seq
            logging.debug("Worker has received sentence %d" % sent_index)
            
            if sent_index == -1:
                logging.info('Received done signal from job server')
                break

            t0 = time.time()
            (self.dyn_prog, log_prob) = self.forward_pass(self.dyn_prog, sent, self.models, self.K, sent_index)
            sent_sample = self.reverse_sample(self.dyn_prog, sent, self.models, self.K, sent_index)
            if sent_index % self.out_freq == 0:
                logging.info("Processed sentence {0}".format(sent_index))

            t1 = time.time()

            parse = PyzmqParse(sent_index, sent_sample, log_prob)
            
            results_socket.send_pyobj(parse)
                      
            if log_prob > 0:
                logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))
        
        ## Tell the sink that i'm quitting:
        results_socket.send_pyobj(PyzmqParse(-1, None, 0))
        logging.debug("Worker disconnecting sockets and finishing up")
#        jobs_socket.disconnect()
#        results_socket.disconnect()