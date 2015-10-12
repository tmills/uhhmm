#!/usr/bin/env python3

class PyzmqJob:
    def __init__(self, index, ev_seq):
        self.index = index
        self.ev_seq = ev_seq

class PyzmqParse:
    def __init__(self, index, state_list, log_prob):
        self.index = index
        self.state_list = state_list
        self.log_prob = log_prob
