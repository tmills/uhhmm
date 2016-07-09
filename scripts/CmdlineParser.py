#!/usr/bin/env python3

import pickle
import HmmParser
import sys
import time
from multiprocessing import Process, Pipe
from State import *

class ParserProcess(Process):
    def __init__(self, model_file, pipe):
        super().__init__()
        self.parser = HmmParser.HmmParser(pickle.load(open(model_file, 'rb') ) )
        self.pipe = pipe
    
    def run(self):
        while True:
            int_tokens = self.pipe.recv()
            if int_tokens == []:
                break
            
            sent_list = self.parser.matrix_parse(int_tokens)
            
            self.pipe.send(sent_list)

def main(args):
    if len(args) < 3:
        sys.stderr.write("Error: 3 required arguments: <models.bin> <dict file> <num threads>\n")
        sys.exit(-1)
    
    model_filename = args[0]
    
    # Read in dictionary so we can map input to integers for parser
    f = open(args[1], 'r')
    word_map = {}
    for line in f:
        #pdb.set_trace()
        (word, index) = line.rstrip().split(" ")
        word_map[word] = int(index)
    
    num_threads = int(args[2])
    procs = []
    pipes = []
    
    for i in range(0, num_threads):
        (worker_pipe, server_pipe) = Pipe()
        procs.append(ParserProcess(model_filename, worker_pipe))
        pipes.append(server_pipe)
        procs[i].start()
        
    proc_ind = -1
    sys.stderr.write("Now accepting sentences from standard input as space-separated tokens:\n")
    t0 = time.time()
    num_tokens = 0
    
    for line in sys.stdin:
        proc_ind += 1
        if proc_ind >= num_threads:
            ## wait for them to finish
            for i in range(0, num_threads):
                sent_list = pipes[i].recv()
                #print("Received output from thread %d with %d states" % (i, len(sent_list)))            
                print(list(map(lambda x: x.str(), sent_list) ) )
                
            proc_ind = 0
        
        str_tokens = line.lower().split()
        num_tokens += len(str_tokens)
        try:
            int_tokens = []
            for token in str_tokens:
                if token in word_map:
                    int_tokens.append(word_map[token])
                else:
                    int_tokens.append(word_map['unk'])

            pipes[proc_ind].send(int_tokens)
                    
#            sent_list = parser.parse(int_tokens)
        except Exception as e:
            sys.stderr.write("Could not parse this sentence due to error %s\n" % (e) )
    
    ## when we quit, ask for the sentences that were sent but never retrieved
    for i in range(0, proc_ind+1):
        sent_list = pipes[i].recv()
        #print("Received output from thread %d with %d states" % (i, len(sent_list)))            
        print(list(map(lambda x: x.str(), sent_list) ) )
        
    ## To quit, give every process the None sentence:
    for i in range(0, num_threads):
        #print("Sending process %d the quit signal" % i)
        pipes[i].send([])
        #print("Waiting for process %d to join" % i)
        procs[i].join()
       
    t1 = time.time()
    print("Total parsing time (not including loading models is %f for %d tokens" % (t1 - t0, num_tokens) )
    time_per_token = (t1-t0) / num_tokens
    corpus_parse_seconds = 226000 * time_per_token
    corpus_parse_minutes = corpus_parse_seconds / 60
    corpus_parse_hours = corpus_parse_minutes / 60
    print("Parser takes %f s per token which would take %f hours to process the whole setE" % (time_per_token, corpus_parse_hours) )
    
if __name__ == "__main__":
    main(sys.argv[1:])
