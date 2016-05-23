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
import pdb, traceback
#import pyximport; pyximport.install()
import DepthOneInfiniteSampler
import HmmSampler
import SparseHmmSampler
from Sampler import *
import Indexer
import FullDepthCompiler

def start_workers(work_distributer, cluster_cmd, maxLen):
    logging.debug("Cluster command is %s" % cluster_cmd)

    cmd_str = 'python3 %s/scripts/PyzmqWorker.py %s %d %d %d %d' % (os.getcwd(), work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen)
    submit_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
    logging.info("Making cluster submit call with the following command: %s" % str(submit_cmd))
    subprocess.call(submit_cmd)
    
class PyzmqWorker(Process):
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, out_freq=100, tid=0, seed=0, level=logging.INFO):
        Process.__init__(self)
        self.host = host
        self.jobs_port = jobs_port
        self.results_port = results_port
        self.models_port = models_port
        self.maxLen = maxLen
        self.out_freq = out_freq
        self.tid = tid
        self.quit = False
        self.seed = seed
        self.debug_level = level
        self.model_file_sig = None
        self.indexer = None
        self.models_socket = None
        self.jobs_socket = None
        self.results_socket = None
        
    def run(self):
        logging.basicConfig(level=self.debug_level)
        context = zmq.Context()
        self.models_socket = context.socket(zmq.REQ)
        self.models_socket.connect("tcp://%s:%s" % (self.host, self.models_port))

        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        self.jobs_socket = context.socket(zmq.REQ)        
        self.jobs_socket.connect("tcp://%s:%s" % (self.host, self.jobs_port))
        
        self.results_socket = context.socket(zmq.PUSH)
        self.results_socket.connect("tcp://%s:%s" % (self.host, self.results_port))

        logging.debug("Worker %d connected to all three endpoints" % self.tid)
                
        while True:
            if self.quit:   
                break

            sampler = None
            #  Socket to talk to server
            logging.debug("Worker %d waiting for new models..." % self.tid)
            self.models_socket.send(b'0')
            msg = self.models_socket.recv_pyobj()
            
            in_file = open(msg, 'rb')
            model_wrapper = pickle.load(in_file)
            in_file.close()
            self.model_file_sig = get_file_signature(msg)
            
            if model_wrapper.model_type == ModelWrapper.HMM:
                sampler = HmmSampler.HmmSampler(self.seed)
                sampler.set_models(model_wrapper.model)
                self.processSentences(sampler)
            elif model_wrapper.model_type == ModelWrapper.INFINITE:
                sampler = DepthOneInfiniteSampler.InfiniteSampler(self.seed)
                sampler.set_models(model_wrapper.model)
                self.processSentences(sampler)
            elif model_wrapper.model_type == ModelWrapper.COMPILE:
                self.indexer = Indexer.Indexer(model_wrapper.model)
                self.processRows(model_wrapper.model)
            else:
                logging.error("Received a model type that I don't know how to process!")

            logging.debug("Worker %d received new models..." % self.tid)

            ## Tell the sink that i'm done:
#            results_socket.send_pyobj(PyzmqParse(-1,None,0))


        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        self.jobs_socket.close()
        self.results_socket.close()
        self.models_socket.close()

    def processRows(self, models):
        while True:
            try:
                ret_val = self.jobs_socket.send_pyobj(self.model_file_sig)
                job = self.jobs_socket.recv_pyobj()
            except Exception as e:
                ## Timeout in the job socket probably means we're done -- quit
                logging.info("Exception thrown while waiting for row to process: %s" % (e) )
                self.quit = True
                break

            if job.type == PyzmqJob.COMPILE:
                compile_job = job.resource
                row = compile_job.index
            elif job.type == PyzmqJob.QUIT:
                break
            else:
                logging.debug("Received unexpected job type while expecting compile job! %s" % job.type)
                raise Exception
            
            (indices, data) = FullDepthCompiler.compile_one_line(len(models.fork), row, models, self.indexer)
            row_output = CompiledRow(row, indices, data)
            self.results_socket.send_pyobj(CompletedJob(PyzmqJob.COMPILE, row_output, True) )
            if row % 10000 == 0:
                logging.info("Compiling row %d" % row)
            
            if self.quit:
                break
                
    def processSentences(self, sampler):
        sampler.initialize_dynprog(self.maxLen)        
    
        sents_processed = 0
    
        if self.quit:
            return

        longest_time = 10
        
        while True: 
            logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
            try:
                ret_val = self.jobs_socket.send_pyobj(self.model_file_sig)
                job = self.jobs_socket.recv_pyobj()
            except Exception as e:
                ## Timeout in the job socket probably means we're done -- quit
                logging.info("Exception raised while waiting for sentence: %s" % (e) )
                self.quit = True
                break
        
            if job.type == PyzmqJob.SENTENCE:
                sentence_job = job.resource
                sent_index = sentence_job.index
                sent = sentence_job.ev_seq
            elif job.type == PyzmqJob.QUIT:
                logging.debug('Worker %d received signal from job server to check for new model' % self.tid)
                break
            elif job.type == PyzmqJob.COMPILE:
                logging.error("Worker %d received compile job from job server when expecting sentence job!")
                raise Exception
            

            logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))                

            t0 = time.time()
        
            success = True
        
            try:
                (sent_sample, log_prob) = sampler.sample(sent, sent_index)
            except Exception as e:
                logging.warning("Warning: Sentence %d had a parsing error %s." % (sent_index, e))
                sent_sample = None
                log_prob = 0
                success = False
        
        
            if sent_index % self.out_freq == 0:
                logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

            t1 = time.time()
        
            if success:
                logging.log(logging.DEBUG-1, "Worker %d has parsed sentence %d with result %s" % (self.tid, sent_index, list(map(lambda x: x.str(), sent_sample))))                
            else:
                logging.info("Worker %d was unsuccessful in attempt to parse sentence %d" % (self.tid, sent_index) )

            if (t1-t0) > longest_time:
                longest_time = t1-t0
                logging.warning("Sentence %d was my slowest sentence to parse so far at %d s" % (sent_index, longest_time) )


            parse = PyzmqParse(sent_index, sent_sample, log_prob, success)
            sents_processed +=1
        
            self.results_socket.send_pyobj(CompletedJob(PyzmqJob.SENTENCE, parse, parse.success))
            if self.quit:
                break

            if log_prob > 0:
                logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))    

        logging.info("Cumulative forward time %f and backward time %f" % (sampler.ff_time, sampler.bs_time))
        logging.debug("Worker %d processed %d sentences this iteration" % (self.tid, sents_processed))


    def handle_sigint(self, signum, frame):
        logging.info("Worker received quit signal... will terminate after cleaning up.")
        self.quit = True
    
    def handle_sigalarm(self, signum, frame):
        logging.warning("Worker received alarm while trying to process sentence... will raise exception")
        raise ParseException("Worker hung while parsing sentence")
        
def main(args):
    logging.basicConfig(level=logging.INFO)
    
    if len(args) != 1 and len(args) != 5:
        print("ERROR: Wrong number of arguments! Two run modes -- One argument of a file with properties or 5 arguments with properties.")
        sys.exit(-1)
        
    if len(args) == 1:
        config_file = args[0] + "/masterConfig.txt"
        while True:
            if os.path.isfile(config_file):
                configs = open(config_file).readlines()
                if len(configs)==2 and 'OK' in configs[1]:
                    logging.info('OSC setup acquired. Starting a worker with ' + config_file)
                    args = configs[0].strip().split(' ')
                    break
            else:
                time.sleep(10)
    
    fs = PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]))
    signal.signal(signal.SIGINT, fs.handle_sigint)
    signal.signal(signal.SIGALRM, fs.handle_sigalarm)
    
    ## Call run directly instead of start otherwise we'll have 2n workers    
    fs.run()
    
if __name__ == "__main__":
    main(sys.argv[1:])

