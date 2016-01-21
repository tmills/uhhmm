#!/usr/bin/env python3

import zmq
from PyzmqMessage import *
from multiprocessing import Process
import logging
import os
import os.path
import pickle
import signal
import subprocess
import sys
import time
import pyximport; pyximport.install()
import DepthOneInfiniteSampler
import HmmSampler
from Sampler import *

def start_workers(work_distributer, cluster_cmd, maxLen):
    logging.debug("Cluster command is %s" % cluster_cmd)

    cmd_str = 'python3 %s/scripts/PyzmqWorker.py %s %d %d %d %d' % (os.getcwd(), work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen)
    submit_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
    logging.info("Making cluster submit call with the following command: %s" % str(submit_cmd))
    subprocess.call(submit_cmd)
    
class PyzmqWorker(Process):
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, out_freq=100):
        Process.__init__(self)
        self.host = host
        self.jobs_port = jobs_port
        self.results_port = results_port
        self.models_port = models_port
        self.maxLen = maxLen
        self.out_freq = out_freq
        self.tid = 0
        self.quit = False

    def run(self):
        logging.debug("Starting forward pass in thread %d", self.tid)

        context = zmq.Context()
        models_socket = context.socket(zmq.REQ)
        models_socket.connect("tcp://%s:%s" % (self.host, self.models_port))

        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        jobs_socket = context.socket(zmq.REQ)        
        jobs_socket.connect("tcp://%s:%s" % (self.host, self.jobs_port))
        jobs_socket.setsockopt(zmq.SNDTIMEO, 5000)
        jobs_socket.setsockopt(zmq.RCVTIMEO, 5000)
        
        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%s" % (self.host, self.results_port))

        logging.debug("Worker %d connected to all three endpoints" % self.tid)
                
        while True:
            if self.quit:   
                break

            sampler = None
            #  Socket to talk to server
            logging.debug("Thread %d waiting for new models..." % self.tid)
            models_socket.send(b'0')
            msg = models_socket.recv_pyobj()
            
            in_file = open(msg, 'rb')
            model_obj = pickle.load(in_file)
            in_file.close()
            model_file_sig = get_file_signature(msg)
            models = model_obj.model
            finite = model_obj.finite
            
            if finite:
                sampler = HmmSampler.HmmSampler()
            else:
                sampler = DepthOneInfiniteSample.InfiniteSampler()
            
            sampler.set_models(models)
                       
            logging.debug("Thread %d received new models..." % self.tid)

            sampler.initialize_dynprog(self.maxLen)        
            
            sents_processed = 0
            
            if self.quit:
                break

            while True: 
                logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
                try:
                    ret_val = jobs_socket.send_pyobj(model_file_sig)
                    job = jobs_socket.recv_pyobj();
                except:
                    ## Timeout in the job socket probably means we're done -- quit
                    logging.info("Worker timed out waiting for job... shutting down.")
                    self.quit = True
                    break
                
                if job.type == PyzmqJob.SENTENCE:
                    sent_index = job.index
                    sent = job.ev_seq
                elif job.type == PyzmqJob.QUIT:
                    logging.debug('Worker %d received done signal from job server' % self.tid)
                    break

                logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))                

                t0 = time.time()
                
                success = True
                try:
                    (sent_sample, log_prob) = sampler.sample(sent, sent_index)
                except:
                    sent_sample = None
                    log_prob = 0
                    success = False
                                    
                if sent_index % self.out_freq == 0:
                    logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

                t1 = time.time()

                sents_processed +=1 
                parse = PyzmqParse(sent_index, sent_sample, log_prob, success)
                
                results_socket.send_pyobj(parse)
                if self.quit:
                    break

                if log_prob > 0:
                    logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))
            
            ## Tell the sink that i'm done:
#            results_socket.send_pyobj(PyzmqParse(-1,None,0))
            logging.debug("Worker %d processed %d sentences this iteration" % (self.tid, sents_processed))

        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()
        models_socket.close()

    def handle_sigint(self, signum, frame):
        logging.info("Worker received quit signal... will terminate after cleaning up.")
        self.quit = True
        
def main(args):
    logging.basicConfig(level=logging.INFO)

    fs = PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]))
    signal.signal(signal.SIGINT, fs.handle_sigint)
    
    ## Call run directly instead of start otherwise we'll have 2n workers    
    fs.run()
    
if __name__ == "__main__":
    main(sys.argv[1:])

