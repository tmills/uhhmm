#!/usr/bin/env python3

import zmq
from PyzmqMessage import *
from multiprocessing import Process
import logging
import pickle
import time
import subprocess
import pyximport; pyximport.install()
import beam_sampler
import finite_sampler
from Sampler import *

class PyzmqWorker(Process):
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, tid, cluster_cmd=None, out_freq=100):
        Process.__init__(self)
        logging.debug("Sampler %d started" % tid)
        self.host = host
        self.jobs_port = jobs_port
        self.results_port = results_port
        self.models_port = models_port
        self.maxLen = maxLen
        self.tid = tid
        self.out_freq = out_freq
        self.cluster_cmd = cluster_cmd
    
    def start(self):
        if self.cluster_cmd == None:
            ## continue up the call chain
            Process.start(self)
        else:
            logging.debug("Making cluster submit call with the following command: %s" % self.cluster_cmd)
            ## call the cluster submission command
            subprocess.call(self.cluster_cmd)
        
    def run(self):
        logging.debug("Starting forward pass in thread %d", self.tid)

        context = zmq.Context()
        models_socket = context.socket(zmq.REQ)
        models_socket.connect("tcp://%s:%s" % (self.host, self.models_port))

        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        jobs_socket = context.socket(zmq.REQ)        
        jobs_socket.connect("tcp://%s:%s" % (self.host, self.jobs_port))
        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%s" % (self.host, self.results_port))

        logging.debug("Worker %d connected to all three endpoints" % self.tid)
        sampler = None
        
        while True:
            #  Socket to talk to server
            logging.debug("Thread %d waiting for new models..." % self.tid)
            models_socket.send(b'0')
            msg = models_socket.recv_pyobj()
            if msg == None:
                break
            else:
                in_file = open(msg.location, 'rb')
                models = pickle.load(in_file)
                in_file.close()
                
            if msg.finite:
                sampler = finite_sampler.FiniteSampler()
            else:
                sampler = beam_sampler.InfiniteSampler()
            
            sampler.set_models(models)
                       
            logging.debug("Thread %d received new models..." % self.tid)

            sampler.initialize_dynprog(self.maxLen)        
            
            sents_processed = 0
            while True: 
                logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
                jobs_socket.send(b'0')
                job = jobs_socket.recv_pyobj();
                
                if job.type == PyzmqJob.SENTENCE:
                    sent_index = job.index
                    sent = job.ev_seq
                elif job.type == PyzmqJob.QUIT:
                    logging.debug('Worker %d received done signal from job server' % self.tid)
                    break

                logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))                

                t0 = time.time()
                
                (sent_sample, log_prob) = sampler.sample(sent, sent_index)
                
                if sent_index % self.out_freq == 0:
                    logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

                t1 = time.time()

                sents_processed +=1 
                parse = PyzmqParse(sent_index, sent_sample, log_prob)
                
                results_socket.send_pyobj(parse)
                          
                if log_prob > 0:
                    logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))
            
            ## Tell the sink that i'm done:
            results_socket.send_pyobj(PyzmqParse(-1,None,0))
            logging.debug("Worker %d processed %d sentences this iteration" % (self.tid, sents_processed))

        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()

def main(args):
    logging.basicConfig(level=logging.INFO)
    fs = PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5]))
    ## Call run directly instead of start otherwise we'll have 2n workers
    fs.run()
    
if __name__ == "__main__":
    main(sys.argv[1:])

