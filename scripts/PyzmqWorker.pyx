#!/usr/bin/env python3

import zmq
from multiprocessing import Process
from PyzmqMessage import get_file_signature, resource_current, ModelWrapper, PyzmqJob, SentenceJob, CompileJob, CompletedJob, PyzmqParse, CompiledRow, SentenceRequest, RowRequest
import multiprocessing
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
cimport HmmSampler
import SparseHmmSampler
import Sampler
cimport Sampler
#from Sampler import *
import Indexer
import FullDepthCompiler
from uhhmm_io import printException, ParsingError

       
cdef class PyzmqWorker:
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, out_freq=100, tid=0, gpu=False, batch_size=8, seed=0, level=logging.INFO):
        #Process.__init__(self)
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
        self.gpu = gpu
        self.batch_size = batch_size
        #print('GPU %s with batch size %d' % (self.gpu, self.batch_size) )

    def __reduce__(self):
        return (PyzmqWorker, (self.host, self.jobs_port, self.results_port, self.models_port, self.maxLen, self.out_freq, self.tid, self.gpu, self.batch_size, self.seed, self.debug_level), None)
        
    def run(self):
        logging.basicConfig(level=self.debug_level)
        context = zmq.Context()
        models_socket = context.socket(zmq.REQ)
        url = "tcp://%s:%d" % (self.host, self.models_port)
        logging.debug("Connecting to models socket at url %s" % (url) )
        models_socket.connect(url)

        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        jobs_socket = context.socket(zmq.REQ)        
        jobs_socket.connect("tcp://%s:%d" % (self.host, self.jobs_port))
        
        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%d" % (self.host, self.results_port))

        logging.debug("Worker %d connected to all three endpoints" % self.tid)
                
        while True:
            if self.quit:   
                break
            # logging.info("GPU for worker is %s" % self.gpu)
            sampler = None
            #  Socket to talk to server
            logging.debug("Worker %d waiting for new models..." % self.tid)
            models_socket.send(b'0')
            msg = models_socket.recv_pyobj()
            # if self.gpu:
            #     msg = msg + '.gpu' # use gpu model for model
            in_file = open(msg, 'rb')
            try:
                model_wrapper = pickle.load(in_file)
            except Exception as e:
                printException()
                raise e
            in_file.close()
            self.model_file_sig = get_file_signature(msg)
            
            if model_wrapper.model_type == ModelWrapper.HMM and not self.gpu:
                sampler = HmmSampler.HmmSampler(self.seed)
                sampler.set_models(model_wrapper.model)
                self.processSentences(sampler, model_wrapper.model[1], jobs_socket, results_socket)
            elif model_wrapper.model_type == ModelWrapper.INFINITE:
                sampler = DepthOneInfiniteSampler.InfiniteSampler(self.seed)
                sampler.set_models(model_wrapper.model)
                self.processSentences(sampler)
            elif model_wrapper.model_type == ModelWrapper.COMPILE:
                self.indexer = Indexer.Indexer(model_wrapper.model)
                self.processRows(model_wrapper.model, jobs_socket, results_socket)
            elif model_wrapper.model_type == ModelWrapper.HMM and self.gpu:
                import CHmmSampler
                msg = msg + '.gpu'
                in_file = open(msg, 'rb') # loading a specific gpu model and trick the system to believe it is the normal model
                model_wrapper = pickle.load(in_file)
                sampler = CHmmSampler.GPUHmmSampler(self.seed)
                # print(model_wrapper.model)
                gpu_model = CHmmSampler.GPUModel(model_wrapper.model)
                sampler.set_models(gpu_model)
                in_file.close()
                # self.model_file_sig = get_file_signature(msg)
                pi = 0 # placeholder
                self.processSentences(sampler, pi, jobs_socket, results_socket)


            else:
                logging.error("Received a model type that I don't know how to process!")

            logging.debug("Worker %d received new models..." % self.tid)

            ## Tell the sink that i'm done:
#            results_socket.send_pyobj(PyzmqParse(-1,None,0))


        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()
        models_socket.close()

    def processRows(self, models, jobs_socket, results_socket):
        while True:
            try:
                ret_val = jobs_socket.send_pyobj(RowRequest(self.model_file_sig))
                job = jobs_socket.recv_pyobj()
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
            results_socket.send_pyobj(CompletedJob(PyzmqJob.COMPILE, row_output, True) )
            if row % 10000 == 0:
                logging.info("Compiling row %d" % row)
            
            if self.quit:
                break
                
    def processSentences(self, sampler, pi, jobs_socket, results_socket):
        batch_size = 1
        sampler.initialize_dynprog(batch_size, self.maxLen)
    
        sents_processed = 0
    
        if self.quit:
            return

        longest_time = 10
        sent_batch = []
        epoch_done = False
        
        while True: 
            logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
            try:
                
                ret_val = jobs_socket.send_pyobj(SentenceRequest(self.model_file_sig, self.batch_size))
                jobs = jobs_socket.recv_pyobj()
                if self.batch_size == 1:
                    job = jobs
                else:
                    job = jobs[0]
            except Exception as e:
                ## Timeout in the job socket probably means we're done -- quit
                logging.info("Exception raised while waiting for sentence: %s" % (e) )
                self.quit = True
                break
            if job.type == PyzmqJob.SENTENCE:
                sentence_job = job.resource
                sent = sentence_job.ev_seq
                sent_batch.append(sent)
                if len(sent_batch) == 1:
                    sent_index = sentence_job.index
            elif job.type == PyzmqJob.QUIT:
                logging.debug('Worker %d received signal from job server to check for new model' % self.tid)
                epoch_done = True
                if len(sent_batch) == 0:
                    ## We got the epoch done signal with no sentences to process
                    break
            elif job.type == PyzmqJob.COMPILE:
                logging.error("Worker %d received compile job from job server when expecting sentence job!")
                raise Exception
            

            logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))                

            t0 = time.time()
        
        
            if len(sent_batch) >= batch_size or epoch_done:
                if batch_size > 1:
                    logging.info("Batch now has %d sentences and size is %d so starting to process" % (len(sent_batch), batch_size) )

                success = True
                t0 = time.time()
                try:
                    (sent_samples, log_probs) = sampler.sample(pi, sent_batch, sent_index)
                except Exception as e:
                    logging.warning("Warning: Sentence %d had a parsing error %s." % (sent_index, e))
                    sent_sample = None
                    log_prob = 0
                    success = False

                if sent_index % self.out_freq == 0:
                    logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

                t1 = time.time()
         
                if not success:
                    logging.info("Worker %d was unsuccessful in attempt to parse sentence %d" % (self.tid, sent_index) )
        
                if batch_size > 1 or (t1-t0) > longest_time:
                    longest_time = t1-t0
                    logging.warning("Sentence %d was my slowest sentence to parse so far at %d s" % (sent_index, longest_time) )

                for ind,sent_sample in enumerate(sent_samples):
                    parse = PyzmqParse(sent_index, sent_sample, log_probs[ind], success)
                    sents_processed +=1        
                    results_socket.send_pyobj(CompletedJob(PyzmqJob.SENTENCE, parse, parse.success))

                sent_batch = []

            if self.quit:
                break

            if log_prob > 0:
                logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))    

        logging.debug("Worker %d processed %d sentences this iteration" % (self.tid, sents_processed))

    def handle_sigint(self, signum, frame):
        logging.info("Worker received quit signal... will terminate after cleaning up.")
        self.quit = True

    def handle_sigterm(self, signum, frame):
        logging.info("Worker received quit signal... will terminate after cleaning up.")
        self.quit = True
    
    def handle_sigalarm(self, signum, frame):
        logging.warning("Worker received alarm while trying to process sentence... will raise exception")
        raise ParsingError("Worker hung while parsing sentence")

                
# def main(args):
#     logging.basicConfig(level=logging.INFO)
    
#     if len(args) != 1 and len(args) != 5:
#         print("ERROR: Wrong number of arguments! Two run modes -- One argument of a file with properties or 5 arguments with properties.")
#         sys.exit(-1)
        
#     if len(args) == 1:
#         config_file = args[0] + "/masterConfig.txt"
#         while True:
#             if os.path.isfile(config_file):
#                 configs = open(config_file).readlines()
#                 if len(configs)==2 and 'OK' in configs[1]:
#                     logging.info('OSC setup acquired. Starting a worker with ' + config_file)
#                     args = configs[0].strip().split(' ')
#                     break
#             else:
#                 time.sleep(10)
    
#     if len(args) == 6:
#         fs = PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), gpu=bool(args[5]))
#     elif len(args) == 5:
#         fs = PyzmqWorker(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]))
#     signal.signal(signal.SIGINT, fs.handle_sigint)
#     signal.signal(signal.SIGALRM, fs.handle_sigalarm)
    
#     ## Call run directly instead of start otherwise we'll have 2n workers    
#     fs.run()
    
# if __name__ == '__main__':
#     args = sys.argv[1:]
#     main(args)
