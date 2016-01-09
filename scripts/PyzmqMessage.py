#!/usr/bin/env python3
# Classes and methods for facilitating communication between workers and server

import os

class PyzmqJob:
    SENTENCE=0
    QUIT=1
    
    def __init__(self, msg_type, index, ev_seq):
        self.type = msg_type
        self.index = index
        self.ev_seq = ev_seq

class PyzmqParse:
    def __init__(self, index, state_list, log_prob, success=True):
        self.index = index
        self.state_list = state_list
        self.log_prob = log_prob
        self.success = success

class PyzmqModel:
    def __init__(self, model, finite):
        self.model = model
        self.finite = finite

def model_current(ref_sig, comp_sig):
    return (ref_sig[0] == comp_sig[0] and ref_sig[1] == comp_sig[1])
    
def get_file_signature(filename):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(filename)
    return (size, mtime)    

    