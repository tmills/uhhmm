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
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, out_freq=100, tid=0, gpu=False, batch_size=8, seed=0, max_runtime_in_seconds=None, level=logging.INFO):
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

        # Sometimes we may want a worker to run for no longer than a specified amount of time
        #
        # The following variables relate to that scenario
        if max_runtime_in_seconds == None:
            self.scheduled_time_of_death = float("inf")
        else:
            self.scheduled_time_of_death = time.time() + max_runtime_in_seconds
        logging.info("Scheduled time of death is {}".format(self.scheduled_time_of_death))
        self.longest_wait_for_new_model = 0
        self.longest_wait_processing_sentences = 0
        self.longest_wait_processing_rows = 0

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
            time_before_new_model = time.time()
            if time_before_new_model + self.longest_wait_for_new_model >= self.scheduled_time_of_death:
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
            self.longest_wait_for_new_model = max(time.time() - time_before_new_model, self.longest_wait_for_new_model)
            
            if model_wrapper.model_type == ModelWrapper.HMM and not self.gpu:
                time_before = time.time()
                if time_before + self.longest_wait_processing_sentences < self.scheduled_time_of_death:
                    sampler = HmmSampler.HmmSampler(self.seed)
                    sampler.set_models(model_wrapper.model)
                    self.processSentences(sampler, model_wrapper.model[1], jobs_socket, results_socket)
                    self.longest_wait_processing_sentences = max(time.time()-time_before, self.longest_wait_processing_sentences)
                else:
                    logging.info("Worker {} will not process sentences due to lack of time before scheduled worker completion:\t{} + {} >= {}".format(self.tid, time_before, self.longest_wait_processing_sentences, self.scheduled_time_of_death))
                    break

            elif model_wrapper.model_type == ModelWrapper.INFINITE:
                time_before = time.time()
                if time_before + self.longest_wait_processing_sentences < self.scheduled_time_of_death:
                    sampler = DepthOneInfiniteSampler.InfiniteSampler(self.seed)
                    sampler.set_models(model_wrapper.model)
                    self.processSentences(sampler)
                    self.longest_wait_processing_sentences = max(time.time()-time_before, self.longest_wait_processing_sentences)
                else:
                    logging.info("Worker {} will not process sentences due to lack of time before scheduled worker completion:\t{} + {} >= {}".format(self.tid, time_before, self.longest_wait_processing_sentences, self.scheduled_time_of_death))
                    break

            elif model_wrapper.model_type == ModelWrapper.COMPILE:
                time_before = time.time()
                if time_before + self.longest_wait_processing_rows < self.scheduled_time_of_death:
                    self.indexer = Indexer.Indexer(model_wrapper.model)
                    self.processRows(model_wrapper.model, jobs_socket, results_socket)
                    self.longest_wait_processing_rows = max(time.time()-time_before, self.longest_wait_processing_rows)
                else:
                    logging.info("Worker {} will not process rows due to lack of time before scheduled worker completion:\t{} + {} >= {}".format(self.tid, time_before, self.longest_wait_processing_rows, self.scheduled_time_of_death))
                    break

            elif model_wrapper.model_type == ModelWrapper.HMM and self.gpu:
                time_before = time.time()
                if time_before + self.longest_wait_processing_sentences < self.scheduled_time_of_death:
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
                    self.longest_wait_processing_sentences = max(time.time()-time_before, self.longest_wait_processing_sentences)
                else:
                    logging.info("Worker {} will not process sentences due to lack of time before scheduled worker completion:\t{} + {} >= {}".format(self.tid, time_before, self.longest_wait_processing_sentences, self.scheduled_time_of_death))
                    break


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
            if time.time() >= self.scheduled_time_of_death:
                logging.info("Worker {} will stop processing rows due to lack of time before scheduled worker completion:\t{} >= {}".format(self.tid, time.time(), self.scheduled_time_of_death))
                self.quit = True
                break

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
        sampler.initialize_dynprog(self.batch_size, self.maxLen)
    
        sents_processed = 0
    
        if self.quit:
            return

        longest_time = 10
        sent_batch = []
        epoch_done = False
        
        while True: 
            if time.time() >= self.scheduled_time_of_death:
                logging.info("Worker {} will stop processing sentences due to lack of time before scheduled worker completion:\t{} >= {}".format(self.tid, time.time(), self.scheduled_time_of_death))
                self.quit = True
                break

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
                if self.batch_size > 1:
                    sent_index = job.resource.index
                    for job in jobs:
                        sentence_job = job.resource
                        sent = sentence_job.ev_seq
                        sent_batch.append(sent)
                else:
                    sentence_job = job.resource
                    sent_index = sentence_job.index
                    sent = sentence_job.ev_seq
                    sent_batch.append(sent)
            elif job.type == PyzmqJob.QUIT:
                logging.info('Worker %d received signal from job server to check for new model' % self.tid)
                epoch_done = True
                if len(sent_batch) == 0:
                    ## We got the epoch done signal with no sentences to process
                    break
            elif job.type == PyzmqJob.COMPILE:
                logging.error("Worker %d received compile job from job server when expecting sentence job!")
                raise Exception
            

            logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))                

            t0 = time.time()
        
        
            if True: # len(sent_batch) >= self.batch_size or epoch_done:
                #if self.batch_size > 1:
                    #logging.info("Batch now has %d sentences and size is %d so starting to process" % (len(sent_batch), self.batch_size) )

                success = True
                t0 = time.time()
                try:
                    if len(sent_batch) in [1,2,4,8,16,32,64,128,256]:
                        #logging.info("Batch has acceptable length of %d" % (len(sent_batch)))
                        (sent_samples, log_probs) = sampler.sample(pi, sent_batch, sent_index)
                    else:
                        logging.info("Batch size %d doesn't match power of 2 -- breaking into sub-batches" % (len(sent_batch) ) )
                        ## have some number of batches < 32 but not a power of 2
                        sent_samples = []
                        log_probs = []
                        
                        for mini_batch_size in (256,128,64,32,16,8,4,2,1):
                            if len(sent_batch) >= mini_batch_size:
                                logging.info("Processing mini-batch of size %d" % (mini_batch_size) )
                                sub_batch = sent_batch[0:mini_batch_size]
                                sent_batch = sent_batch[mini_batch_size:]
                               
                                try:
                                    (sub_samples, sub_probs) = sampler.sample(pi, sub_batch, sent_index)
                                except e as Exception:
                                    print("Exception in sampler: %s" % (str(e)))
                                    raise Exception
                                    
                                sent_samples.extend(sub_samples)
                                log_probs.extend(sub_probs)
                                
                        logging.info("After chopping up final batch, we have %d samples processed." % (len(sent_samples)))
                except Exception as e:
                    logging.warning("Warning: Sentence %d had a parsing error %s." % (sent_index, e))
                    sent_sample = None
                    log_prob = 0
                    success = False

                if (self.batch_size == 1 and sent_index % self.out_freq == 0) or (self.batch_size > 1 and (sent_index // self.out_freq != (sent_index+len(sent_batch)) // self.out_freq)):
                    logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

                t1 = time.time()
         
                if not success:
                    logging.info("Worker %d was unsuccessful in attempt to parse sentence %d" % (self.tid, sent_index) )
        
                if self.batch_size == 1 and (t1-t0) > longest_time:
                    longest_time = t1-t0
                    logging.warning("Sentence %d was my slowest sentence to parse so far at %d s" % (sent_index, longest_time) )

                for ind,sent_sample in enumerate(sent_samples):
                    parse = PyzmqParse(sent_index+ind, sent_sample, log_probs[ind], success)
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
