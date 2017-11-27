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
import tempfile
import socket
import DepthOneInfiniteSampler
import HmmSampler
cimport HmmSampler
import SparseHmmSampler
import Sampler
cimport Sampler
#from Sampler import *
import Indexer
import FullDepthCompiler
import os
import time
from uhhmm_io import printException, ParsingError
import CategoricalObservationModel
import GaussianObservationModel
import models
from WorkDistributerServer import get_local_ip
import CHmmSampler
import threading

## This function was required because of some funkiness on ubuntu systems where reverse dns lookup was returning a loopback ip
## This will try the easy way and if it returns something with 127. will make an outside connectino to known DNS (8.8.8.8) and
## grab the external IP from that socket.
## Solution from here: http://stackoverflow.com/a/1267524
def get_local_ip():
    return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

cdef class PyzmqWorker:
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, out_freq=100, tid=0, gpu=False, batch_size=8, seed=0, level=logging.INFO):
        #Process.__init__(self)
        logging.info("Thread created with id %s" % (threading.get_ident()))
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
        self.my_ip = get_local_ip()

    def __reduce__(self):
        return (PyzmqWorker, (self.host, self.jobs_port, self.results_port, self.models_port, self.maxLen, self.out_freq, self.tid, self.gpu, self.batch_size, self.seed, self.debug_level), None)

    def run(self):
        logging.basicConfig(level=self.debug_level)
        logging.info("Thread starting run() method with id=%s" % (threading.get_ident()))
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
            #  Socket to talk to server
            logging.debug("Worker %d sending request for new model" % self.tid)
            models_socket.send(b'0')
            logging.debug("Worker %d waiting for new model location from server." % self.tid)
            msg = models_socket.recv_pyobj()
            logging.debug("Worker %d received new model location from server." % self.tid)
            # if self.gpu:
            #     msg = msg + '.gpu' # use gpu model for model

            try:
                model_wrapper, self.model_file_sig = self.get_model(msg)
            except Exception as e:

                printException()
                raise e

            logging.debug("Worker %d preparing to process new model" % self.tid)

            if model_wrapper.model_type == ModelWrapper.HMM and not self.gpu:
                if self.batch_size > 0:
                    logging.info("Worker %d doing finite model inference on the CPU" % (self.tid))
                    #print("Observation model type is %s" % (type(model_wrapper.model[0].lex)))
                    if isinstance(model_wrapper.model[0].lex, models.CategoricalModel):
                        obs_model = CategoricalObservationModel.CategoricalObservationModel()
                    else:
                        obs_model = GaussianObservationModel.GaussianObservationModel()

                    sampler = HmmSampler.HmmSampler(seed=self.seed, obs_model=obs_model)
                    sampler.set_models(model_wrapper.model)
                    self.processSentences(sampler, model_wrapper.model[1], jobs_socket, results_socket)
                else:
                    time.sleep(2)

            elif model_wrapper.model_type == ModelWrapper.INFINITE:
                logging.info("Worker %d doing infinite model inference" % (self.tid))
                sampler = DepthOneInfiniteSampler.InfiniteSampler(self.seed)
                sampler.set_models(model_wrapper.model)
                self.processSentences(sampler, model_wrapper.model, jobs_socket, results_socket)

            elif model_wrapper.model_type == ModelWrapper.COMPILE:
                logging.info("Worker %d in compile stage" % (self.tid))
                self.indexer = Indexer.Indexer(model_wrapper.model)
                self.processRows(model_wrapper.model, jobs_socket, results_socket, depth_limit=model_wrapper.depth)

            elif model_wrapper.model_type == ModelWrapper.HMM and self.gpu:
                logging.info("Worker %d doing finite model inference on the GPU" % (self.tid))
                import CHmmSampler
                msg.file_path = msg.file_path + '.gpu'
                lex = model_wrapper.model[0].lex
                model_wrapper = self.get_model(msg)[0]

                # print('1 worker loading file.')
                # print('2 init sampler on GPU')
                # print(model_wrapper.model)
                if isinstance(lex, models.CategoricalModel):
                    logging.info("Creating gpu sampler with categorical observation model.")
                    sampler = CHmmSampler.GPUHmmSampler(self.seed, CHmmSampler.ModelType.CATEGORICAL_MODEL)
                else:
                    logging.info("Creating gpu sampler with gaussian observation model.")
                    sampler = CHmmSampler.GPUHmmSampler(self.seed, CHmmSampler.ModelType.GAUSSIAN_MODEL)

                gpu_model = CHmmSampler.GPUModel(model_wrapper.model)

                # print('3 loading in models')
                sampler.set_models(gpu_model)
                # print('4 setting in models')
                # self.model_file_sig = get_file_signature(msg)
                pi = 0 # placeholder
                self.processSentences(sampler, pi, jobs_socket, results_socket)
                ## Free memory of gpu model
                model_wrapper = None

            else:
                logging.error("Received a model type that I don't know how to process!")

            logging.debug("Worker %d received new models..." % self.tid)


        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()
        models_socket.close()

    def processRows(self, models, jobs_socket, results_socket, depth_limit=-1):
        if depth_limit == -1:
            depth = len(models.fork)
        else:
            depth = depth_limit

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
                full_pi = compile_job.full_pi
            elif job.type == PyzmqJob.QUIT:
                break
            else:
                logging.debug("Received unexpected job type while expecting compile job! %s" % job.type)
                raise Exception

            (indices, data, indices_full, data_full) = FullDepthCompiler.compile_one_line(depth, row, models, self.indexer, full_pi)
            row_output = CompiledRow(row, indices, data, indices_full, data_full)
            results_socket.send_pyobj(CompletedJob(PyzmqJob.COMPILE, row_output, True) )

            if self.quit:
                break

    def processSentences(self, sampler, pi, jobs_socket, results_socket):
        sampler.initialize_dynprog(self.batch_size, self.maxLen)
        # print('5 init dynprog with batch_size %d and maxlen %d' % (self.batch_size, self.maxLen))

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

                success = False
                t0 = time.time()
                tries = 0

                while not success and tries < 10:
                    try:
                        if tries > 0:
                            logging.info("Error in previous sampling attempt. Retrying batch")
                        if len(sent_batch) in [1,2,4,8,16,32,64,128,256]:
                            #logging.info("Batch has acceptable length of %d" % (len(sent_batch)))
                            # print('6 sampling now')
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
                                    except Exception as e:
                                        print("Exception in sampler: %s" % (str(e)))
                                        raise Exception
                                    sent_samples.extend(sub_samples)
                                    log_probs.extend(sub_probs)

                            logging.info("After chopping up final batch, we have %d samples processed." % (len(sent_samples)))
                        success = True

                    except Exception as e:
                        logging.warning("Warning: Sentence %d had a parsing error %s." % (sent_index, e))
                        tries += 1
                        sent_sample = None
                        log_prob = 0

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

    def get_model(self, model_loc):
        ip = model_loc.ip_addr
        if ip == self.my_ip or ip.startswith('10.'):
            in_file = open(model_loc.file_path, 'rb')
            file_sig = get_file_signature(model_loc.file_path)
        else:
            dir = tempfile.mkdtemp()
            local_path = os.path.join(dir, os.path.basename(model_loc.file_path))
            logging.info("Model location is remote... ssh-ing into server to get model file %s and saving to %s" % (model_loc.file_path, local_path))
            os.system("scp -p %s:%s %s" % (model_loc.ip_addr, model_loc.file_path, local_path))
            in_file = open(local_path, 'rb')
            file_sig = get_file_signature(local_path)
        while True:
            try:
                model = pickle.load(in_file)
                break
            except EOFError:
                logging.warning("EOF error encounter at model loading")
            time.sleep(5)

        in_file.close()
        return model, file_sig

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
