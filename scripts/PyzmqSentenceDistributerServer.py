#!/usr/bin/env python3

import logging
import pickle
import socket
import time
import zmq
from PyzmqMessage import *
from threading import Thread, Lock

class VerboseLock():
    def __init__(self, name):
        self.name = name
        self._lock = Lock()
        
    def acquire(self):
        logging.debug("Acquiring %s lock" % self.name)
        t0 = time.time()
        self._lock.acquire()
        t1 = time.time()
        if t1-t0 > 1:
            logging.warning("Acquiring %s lock took %d seconds" % (self.name, t1-t0))
            
    def release(self):
        logging.debug("Releasing %s lock" % self.name)
        self._lock.release()
            
class Ventilator(Thread):
    def __init__(self, host, sync_port, sent_list):
        Thread.__init__(self)
        self.host = host
        self.sent_list = sent_list
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
            sync = self.sync_socket.recv_pyobj()

            if sync == b'0':
                break
            else:
                current_model_sig = sync

            logging.debug("Ventilator received model signature sync signal")

            for ind,sent in enumerate(self.sent_list):
                logging.log(logging.DEBUG-1, "Ventilator pushing job %d" % ind)
                
                
                while not model_current(current_model_sig, self.socket.recv_pyobj()):
                    ## if the model is not current tell this worker to quit this iteration
#                    logging.log(logging.DEBUG-1, "Current sig is %s, received sig is %s" % (current_model_sig, 
                    self.socket.send_pyobj(PyzmqJob(PyzmqJob.QUIT, -1, None))
    
                ## We heard from a worker with an up to date model, so send it a real sentence
                self.socket.send_pyobj(PyzmqJob(PyzmqJob.SENTENCE, ind, sent))
                logging.log(logging.DEBUG-1, "Ventilator has pushed job %d" % ind)
                
            logging.debug("Ventilator iteration finishing")
        
        logging.debug("Ventilator thread finishing")
        self.socket.close()
        self.sync_socket.close()
        logging.debug("All ventilator sockets closed.")
        
class Sink(Thread):
    def __init__(self, host, sync_port, num_sents):
        Thread.__init__(self)
        self.host = host
        self.num_sents = num_sents
        self.outputs = list()
        logging.debug("Parse accumulator attempting to bind to PULL socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)

        self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        
        logging.debug("Parse accumulator successfully bound to PULL socket.")
        
        self.sync_socket = context.socket(zmq.REQ)
        self.sync_socket.connect("tcp://%s:%s" % (self.host, sync_port))
        logging.debug("Sink connected to sync socket.")
        
        self.work_lock = VerboseLock("Sink")
        self.processing = False

    def run(self):
    
        while True:
            logging.debug("Sink waiting for permission to start...")
            self.sync_socket.send(b'0')
            sync = self.sync_socket.recv_pyobj()
            if sync == b'0':
                break

            model_sig = sync
            
            logging.debug("Sink received model signature sync signal")
            
            num_done = 0
            self.outputs = list()
                  
            while len(self.outputs) < self.num_sents:
                try:   
                    parse = self.socket.recv_pyobj()
                    logging.log(logging.DEBUG-1, "Sink received parse %d" % parse.index)
                    self.outputs.append(parse)
                except:
                    logging.error("Sink timed out waiting for remaining parse... exiting this iteration...")
                    break

            logging.debug("Sink finished processing this batch of sentences")

            self.setProcessing(False)

        logging.debug("Sink thread finishing")
        self.socket.close()
        self.sync_socket.close()
        logging.debug("All sink sockets closed.")
        self.setProcessing(False)

    def setProcessing(self, val):
        self.work_lock.acquire()
        self.processing = val
        self.work_lock.release()

    def getProcessing(self):
        val = self.processing
        return val

    def get_parses(self):
        return self.outputs

class ModelDistributer(Thread):
    def __init__(self, host, sync_port, working_dir):
        Thread.__init__(self)
        self.host = host
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.quit_socket = context.socket(zmq.REQ)
        
        ## disconnect every TO ms to check for quit flag.
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        self.quit_socket.connect("tcp://%s:%s" % (self.host, self.port))
        
        logging.debug("Model server successfully bound to REP socket")
        self.working_dir = working_dir
        self.model_sig = None
        self.model_lock = VerboseLock("Model")
        self.quit = False
        
    ## All this method does is wait for requests for the model and send them,
    ## with a quick check to make sure that the model isn't currently being written
    def run(self):
        model_loc = self.working_dir + '/models.bin'
        
        ## Wait until we're actually given a model to start sending them out...
        while self.model_sig == None:
            time.sleep(1)

        while True:
            ## TODO -- put a timeout in this recv() of a second so that it checks the
            ## quit flag regularly
            try:
                sync = self.socket.recv()
                if sync == b'-1':
                    logging.info("Model server received quit signal")
                    break
                logging.log(logging.DEBUG, 'Sending worker a model in response to signal %s' % str(sync))
                self.model_lock.acquire()
                self.socket.send_pyobj(model_loc)
                self.model_lock.release()
                ## Don't need to do anything -- this happens when there is a timeout,
                ## and just need to check the quit value regularly. Don't know when
                ## to quit otherwise because we can't be sure of how many workers there
                ## are and that they've all quit.
            except e:
                logging.debug("Model server exception %s... checking quit flag." % str(e))
                raise

        self.socket.close()

    def send_models(self, models, finite):
        fn = self.working_dir+'/models.bin'
        self.model_lock.acquire()
        out_file = open(fn, 'wb')
        model = PyzmqModel(models, finite)
        pickle.dump(model, out_file)
        out_file.close()
        self.model_sig = get_file_signature(fn)
        self.model_lock.release()

    def send_quit(self):
        self.quit_socket.send(b'-1')
            
class PyzmqSentenceDistributerServer():
   
    def __init__(self, sent_list, working_dir):

        ## Set up job distribution servers:
        self.host = socket.gethostbyname(socket.gethostname())
        self.sent_list = sent_list

        context = zmq.Context()

        self.sync_socket = context.socket(zmq.REP)
        sync_port = self.sync_socket.bind_to_random_port("tcp://"+self.host)

        self.vent = Ventilator(self.host, sync_port, sent_list)
        self.sink = Sink(self.host, sync_port, len(sent_list))
        self.model_server = ModelDistributer(self.host, sync_port, working_dir)
        
        self.jobs_port = self.vent.port
        self.results_port = self.sink.port
        self.models_port = self.model_server.port

        self.sink.start()
        self.vent.start()
        
        self.sink_socket = context.socket(zmq.PUSH)
        self.sink_socket.connect("tcp://%s:%s" % (self.sink.host, self.sink.port))
        
        self.models = None
        self.model_server.start()
        
    def run_one_iteration(self, models, finite):
        ind = 0
        num_done = 0
        
        self.model_server.send_models(models, finite)
        model_sig = self.model_server.model_sig
        
        self.sink.setProcessing(True)
        
        ## Wait a bit for sink to process signal and set processing to true for the first time
        time.sleep(3)

        ## Wait for the sink to be ready before we start sending sentences out:
        logging.debug("Sending new model signatures %s to threads:" % str(model_sig))
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(model_sig)
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(model_sig)
        
        while self.sink.getProcessing():
            time.sleep(2)

        logging.debug("Sentence distributer server finished with one iteration.")

    def stop(self):
        ## Send two stop signals to sink and ventilator
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(b'0')
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(b'0')
        
        self.sync_socket.close()
        self.sink_socket.close()

        self.model_server.send_quit()
        
        logging.debug("Joining server threads.")
        self.sink.join()
        self.vent.join()
        self.model_server.join()
        logging.debug("All threads joined and exiting server.")
        
    def get_parses(self):
        return self.sink.get_parses()

