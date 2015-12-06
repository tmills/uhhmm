#!/usr/bin/env python3

class PyzmqJob:
    SENTENCE=0
    QUIT=1
    
    def __init__(self, msg_type, index, ev_seq):
        self.type = msg_type
        self.index = index
        self.ev_seq = ev_seq

class PyzmqParse:
    def __init__(self, index, state_list, log_prob):
        self.index = index
        self.state_list = state_list
        self.log_prob = log_prob

class PyzmqModel:
    def __init__(self, location, finite):
        self.location = location
        self.finite = finite
