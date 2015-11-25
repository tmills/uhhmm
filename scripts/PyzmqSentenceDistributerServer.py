#!/usr/bin/env python3

import logging
import pickle
import socket
import time
import zmq
from PyzmqMessage import *
from threading import Thread, Lock

class Ventilator(Thread):
    def __init__(self, host, sync_port, sent_list, num_workers):
        Thread.__init__(self)
        self.host = host
        self.sent_list = sent_list
        self.num_workers = num_workers
        logging.debug("Job distributer attempting to bind to PUSH socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)

        logging.debug("Ventilator successfully bound to PUSH socket.")
            
        self.sync_socket = context.socket(zmq.REQ)
        self.sync_socket.connect("tcp://%s:%s" % (self.host, sync_port))
        logging.debug("Ventilator connected to sync socket.")
        
    def run(self):
        while True:
            ## Wait for signal to start:
            logging.debug("Ventilator waiting for permission to start")
            self.sync_socket.send(b'0')
            sync = self.sync_socket.recv()
            if sync == b'0':
                break

            logging.debug("Ventilator received sync signal %s and starting" % sync)

            for ind,sent in enumerate(self.sent_list):
                logging.log(logging.DEBUG-1, "Ventilator pushing job %d" % ind)
                self.socket.recv()
                self.socket.send_pyobj(PyzmqJob(PyzmqJob.SENTENCE, ind, sent))
                logging.log(logging.DEBUG-1, "Ventilator has pushed job %d" % ind)
                
            for i in range(0, self.num_workers):
                self.socket.recv()
                self.socket.send_pyobj(PyzmqJob(PyzmqJob.QUIT, -1, None))

            time.sleep(1)
            logging.debug("Ventilator iteration finishing")
        
        logging.debug("Ventilator thread finishing")
        self.socket.close()
        self.sync_socket.close()
        
class Sink(Thread):
    def __init__(self, host, sync_port, num_workers):
        Thread.__init__(self)
        self.host = host
        self.outputs = list()
        logging.debug("Parse accumulator attempting to bind to PULL socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.num_workers = num_workers
        
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        
        logging.debug("Parse accumulator successfully bound to PULL socket.")
        
        self.sync_socket = context.socket(zmq.REQ)
        self.sync_socket.connect("tcp://%s:%s" % (self.host, sync_port))
        logging.debug("Sink connected to sync socket.")
        
        self.work_lock = Lock()
        self.processing = False

    def run(self):
    
        while True:
            logging.debug("Sink waiting for permission to start...")
            self.sync_socket.send(b'0')
            sync = self.sync_socket.recv()
            if sync == b'0':
                break

            logging.debug("Sink received start bit %s" % sync)
            num_done = 0
            self.outputs = list()
                        
            while num_done < self.num_workers:
                parse = self.socket.recv_pyobj()
                logging.log(logging.DEBUG-1, "Sink received parse %d" % parse.index)
                if parse.index == -1:
                    num_done += 1
                else:
                    self.outputs.append(parse)
        
            self.setProcessing(False)

        logging.debug("Sink thread finishing")
        self.socket.close()
        self.sync_socket.close()

    def setProcessing(self, val):
        self.work_lock.acquire()
        self.processing = val
        self.work_lock.release()

    def getProcessing(self):
        self.work_lock.acquire()
        val = self.processing
        self.work_lock.release()
        return val

    def get_parses(self):
        return self.outputs

class ModelDistributer():
    def __init__(self, host, sync_port, num_workers, working_dir):
        self.host = host
        self.num_workers = num_workers
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        logging.debug("Model server successfully bound to REP socket")
        self.working_dir = working_dir
        
    def send_models(self, models):
        fn = self.working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        pickle.dump(models, out_file)
        out_file.close()
        for i in range(0, self.num_workers):
            logging.log(logging.DEBUG, 'Sending worker a model')
            self.socket.recv()
            logging.log(logging.DEBUG-1, 'Received signal to send model')
            self.socket.send_pyobj(self.working_dir+'/models.bin')

        logging.debug("Model distributer finished sending models.")

    def send_quit(self):
        for i in range(0, self.num_workers):
            self.socket.recv()
            self.socket.send_pyobj(None)
            
class PyzmqSentenceDistributerServer():
    def __init__(self, sent_list, num_workers, working_dir):

        ## Set up job distribution servers:
        self.host = socket.gethostbyname(socket.gethostname())
        self.sent_list = sent_list

        context = zmq.Context()

        self.sync_socket = context.socket(zmq.REP)
        sync_port = self.sync_socket.bind_to_random_port("tcp://"+self.host)

        self.vent = Ventilator(self.host, sync_port, sent_list, num_workers)
        self.sink = Sink(self.host, sync_port, num_workers)
        self.model_server = ModelDistributer(self.host, sync_port, num_workers, working_dir)
        
        self.jobs_port = self.vent.port
        self.results_port = self.sink.port
        self.models_port = self.model_server.port

        self.sink.start()
        self.vent.start()
        
        self.sink_socket = context.socket(zmq.PUSH)
        self.sink_socket.connect("tcp://%s:%s" % (self.sink.host, self.sink.port))

        self.models = None
        
    def run_one_iteration(self, models):
        ind = 0
        num_workers = 0
        num_done = 0
        
        self.model_server.send_models(models)
        self.sink.setProcessing(True)
        
        ## Wait a bit for sink to process signal and set processing to true for the first time
        time.sleep(3)

        ## Wait for the sink to be ready before we start sending sentences out:
        logging.debug("Sending start bits to threads:")
        self.sync_socket.recv()
        self.sync_socket.send(b'1')
        self.sync_socket.recv()
        self.sync_socket.send(b'1')
        
        while self.sink.getProcessing():
            time.sleep(1)

        logging.debug("Sentence distributer server finished with one iteration.")

    def stop(self):
        ## Send two stop signals to sink and ventilator
        self.sync_socket.recv()
        self.sync_socket.send(b'0')
        self.sync_socket.recv()
        self.sync_socket.send(b'0')
        
        self.model_server.send_quit()
        
    def get_parses(self):
        return self.sink.get_parses()

