#!python3

import numpy as np

def sentence_string(state_list):
    return str(list(map(lambda x: x.str(), state_list)))

# The set of random variable values at one word
# There will be one of these for every word in the training set
class State:
    def __init__(self, int depth, state=None):
        self.depth = depth
        
        if state == None:
            self.f = -1
            self.j = -1
            self.a = np.zeros(depth, dtype=int)
            self.b = np.zeros(depth, dtype=int)
            self.g = 0
        else:
            (self.f, self.j, self.a, self.b, self.g) = state.f, state.j, state.a, state.b, state.g

    def str(self):
        string = ''

        f_str = '+/' if self.f==1 else '-/'        
        string += f_str
        j_str = '+' if self.j==1 else '-'
        string += j_str

        string += "::"
        
        for d in range(0, self.depth):

            if self.a[d] > 0 or d == 0:
                if d > 0:
                    string += ";"
                string += "ACT" + str(self.a[d]) + '/AWA' + str(self.b[d])
        
        string += ':POS' + str(self.g)
        
        return string

    def to_list(self):
        return (self.f, self.j, self.a, self.b, self.g)

    def max_fork_depth(self):
        return self.max_awa_depth()

    def max_awa_depth(self):
        ## Essentially empty -- used for first time step
        if self.b[0] == 0:
            return -1
        
        ## If we encounter a zero at position 1, then the depth is 0
        for d in range(1, self.depth):
            if self.b[d] == 0:
                return d-1
        
        ## Stack is full -- if d=4 then max depth index is 3
        return self.depth-1

