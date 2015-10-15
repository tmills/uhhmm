#!/usr/bin/env python3

import logging
import time
import zmq
from PyzmqMessage import *
from threading import Thread

class Ventilator(Thread):
    def __init__(self, host, port, models, sent_list, num_workers):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.models = models
        self.sent_list = sent_list
        self.num_workers = num_workers
        logging.debug("Job distributer attempting to bind to PUSH socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        if self.port == -1:
            self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        else:
            self.socket.bind("tcp://%s:%s" % (self.host, self.port))

        logging.debug("Job distributer successfully bound to PUSH socket.")
        
    def run(self):        
        for ind,sent in enumerate(self.sent_list):
            logging.log(logging.DEBUG-1, "Ventilator pushing job %d" % ind)
            self.socket.send_pyobj(PyzmqJob(ind, sent))
            logging.log(logging.DEBUG-1, "Ventilator has pushed job %d" % ind)
            
        for i in range(0, self.num_workers):
            self.socket.send_pyobj(PyzmqJob(-1, None))

        time.sleep(1)
        logging.debug("Ventilator run() finishing")
        self.socket.close()
        
class Sink(Thread):
    def __init__(self, host, port, num_workers):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.outputs = list()
        logging.debug("Parse accumulator attempting to bind to PULL socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.num_workers = num_workers
        
        if self.port == -1:
            self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        else:
            self.socket.bind("tcp://%s:%s" % (self.host, self.port))
        
        logging.debug("Parse accumulator successfully bound to PULL socket.")
        
    def run(self):
        start_bit = self.socket.recv()
        logging.debug("Sink received start bit")
        num_done = 0
        
        while num_done < self.num_workers:
            parse = self.socket.recv_pyobj()
            logging.log(logging.DEBUG-1, "Sink received parse %d" % parse.index)
            if parse.index == -1:
                num_done += 1
            else:
                self.outputs.append(parse)
    
        time.sleep(1)
        logging.debug("Sink run() finishing")
        self.socket.close()

    def get_parses(self):
        return self.outputs
            
class PyzmqSentenceDistributerServer(Thread):
    def __init__(self, models, sent_list, num_workers, host="127.0.0.1", jobs_port=-1, results_port=-1):
        Thread.__init__(self)
        self.sent_list = sent_list
        self.models = models
        self.vent = Ventilator(host, jobs_port, models, sent_list, num_workers)
        self.sink = Sink(host, results_port, num_workers)
        self.host = host
        self.jobs_port = self.vent.port
        self.results_port = self.sink.port
        
    def run(self):
        ind = 0   
        num_workers = 0
        num_done = 0
                
        self.sink.start()
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect("tcp://%s:%s" % (self.sink.host, self.sink.port))
        tracker = socket.send(b'0', copy=False, track=True)
        
        while not tracker.done:
            time.sleep(1)

        self.vent.start()
        
        
        logging.debug("Waiting for ventilator to finish")
        self.vent.join()
        
        logging.debug("Waiting for sink to finish")
        self.sink.join()
        
        logging.debug("Sentence distributer server finishing")

    def get_parses(self):
        return self.sink.get_parses()