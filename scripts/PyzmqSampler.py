#!/usr/bin/env python3

import zmq
from PyzmqMessage import *
from multiprocessing import Process
import logging
import time


def getVariableMaxes(models):
    a_max = models.act.dist.shape[-1]
    b_max = models.start.dist.shape[-1]
    g_max = models.pos.dist.shape[-1]
    return (a_max, b_max, g_max)

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
        logging.debug("Starting forward pass in thread %d", self.tid)

        context = zmq.Context()

        #  Socket to talk to server
        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        jobs_socket = context.socket(zmq.PULL)        
        jobs_socket.connect("tcp://%s:%s" % (self.host, self.jobs_port))

        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%s" % (self.host, self.results_port))
        
        logging.debug("Worker %d connected to both endpoints" % self.tid)
        
        while True: 
            logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
            job = jobs_socket.recv_pyobj();
            sent_index = job.index
            sent = job.ev_seq
            logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))
            
            if sent_index == -1:
                logging.debug('Worker %d received done signal from job server' % self.tid)
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
        results_socket.send_pyobj(PyzmqParse(-1,None,0))
        #time.sleep(1)
        
        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()