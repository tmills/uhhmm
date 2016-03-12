#!python3.4
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1

cimport cython
import numpy as np
import sys, os, linecache
import Sampler
import finite_sampler
from ihmm import State
import time
import logging

class ImportanceSet:
    """ A class that represents a finite number of elements according to an importance 
        function. It is initialized with a size and a function that can be called on the
        elements that it will contain. When add() is called, it will check whether the
        set is full and if so, whether the new element is greater than the existing minimum
        element."""
    def __init__(self, size, importance_function, min_value = 0):
        self.size = size
        self.imp_fun = importance_function
        self.minimum_value = min_value
        self.minimum_ind = -1
        self.elements = []
        self.current = 0
        
    def add(self, element):
        if len(self.elements) < self.size:
            self.elements.append(element)
            if len(self.elements) == 1 or self.imp_fun(element) < self.minimum_value:
                self.minimum_value = self.imp_fun(element)
                self.minimum_ind = len(self.elements) - 1
        elif self.imp_fun(element) > self.minimum_value:
            ## Swap out element with minimum value for new element
            self.elements[self.minimum_ind] = element            
            
            ## Set mins to new element (index is already set via above)
            self.minimum_value = self.imp_fun(element)
            
            ## Check for element with minimum value
            for i in range(self.size):
                if self.imp_fun(self.elements[i]) < self.minimum_value:
                    self.minimum_value = self.imp_fun(self.elements[i])
                    self.minimum_ind = i        
        
        ## Otherwise ignore it
        
    def __iter__(self):
        return iter(self.elements)
    
class ParserCell:
    def __init__(self):
        self.prob = 0
        self.state_index = -1
        self.bp = None

class HmmParser:
    def __init__(self, models):
        (self.models, self.pi) = models.model
        self.lil_trans = self.pi.tolil()
        unlog_models(self.models)
        self.totalK = Sampler.get_state_size(self.models)
        self.maxes = Sampler.getVariableMaxes(self.models)

    def parse(self, sent):
        self.lexMatrix = np.matrix(self.models.lex.dist, copy=False)
        (a_max, b_max, g_max) = self.maxes
        
        forward = []
        try:
            for index,token in enumerate(sent):
                t0 = time.time()
                print("Index=%d" % (index))
                ## Still use special case for 0
                if index == 0:
                    cur_probs = ImportanceSet(100, lambda x: x.prob)
                    for curState in range(0,g_max):
                        cell = ParserCell()
                        if curState >= 1:
                            cell.prob = self.lexMatrix[curState, token]
                            cell.bp = None
                            cell.state_index = curState                            
                            cur_probs.add(cell)
                    forward.append(cur_probs)
                else:
                    cur_probs = ImportanceSet(50, lambda x: x.prob)
                    normalizer = 0
                    for curState in range(self.totalK):
                        ## Create the parser cell for this state:
                        max_trans = 0
                        max_bp = None
                        
                        
                        ## Find the input with the highest existing * transition prob:
                        for prevState in forward[-1]:
                            trans_prob = prevState.prob * float(self.lil_trans[prevState.state_index,curState])
                            #print("Trans prob type = %s, shape=%s, prevState index=%d, curState=%d, lil matrix shape = %s" % (type(trans_prob), trans_prob.shape, prevState.state_index, curState, self.lil_trans.shape))
                            ## See if the transition with highest probability
                            if trans_prob > max_trans:
                                max_trans = trans_prob
                                max_bp = prevState
                                
                        if max_trans > 0:
                            ## Extract the pos index for this state so we can get the output prob:
                            (f,j,a,b,g) = finite_sampler.extractStates(curState, self.totalK, self.maxes)
                            max_trans *= self.models.lex.dist[g][token]
                            cur_cell = ParserCell()
                            cur_cell.state_index = curState
                            cur_cell.prob = max_trans
                            cur_cell.bp = max_bp
                            cur_probs.add(cur_cell)                            
                            normalizer += cur_cell.prob
                        
                    forward.append(cur_probs)
                
                t1 = time.time()
                logging.info("Token %d took %f s" % (index, (t1-t0)) )
            ## Find the maximum value that is at depth 0:
            max_prob_cell = None
            max_prob_state = None
            for cell in forward[-1]:
                state = State(finite_sampler.extractStates(cell.state_index, self.totalK, self.maxes))
                if (max_prob_cell == None or cell.prob > max_prob_cell.prob):
                    max_prob_cell = cell
                    max_prob_state = state
            
            reverse_cell_list = [max_prob_cell]
            reverse_sent_list = [state]
            
            ## Backtrack through the sentence
            while reverse_cell_list[-1].bp != None:
                cell = reverse_cell_list[-1].bp
                reverse_cell_list.append(cell)
                reverse_sent_list.append( State(finite_sampler.extractStates(cell.state_index, self.totalK, self.maxes) ) )

            reverse_sent_list.reverse()
                
        except Exception as e:
            printException()
            raise e
        
        return reverse_sent_list    

def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

def unlog_models(models):
    models.fork.dist = 10**models.fork.dist
    
    models.reduce.dist = 10**models.reduce.dist
#    models.trans.dist = 10**models.trans.dist
    
    models.act.dist = 10**models.act.dist
    models.root.dist = 10**models.root.dist
    
    models.cont.dist = 10**models.cont.dist
#    models.exp.dist = 10**models.exp.dist
#    models.next.dist = 10**models.next.dist
    models.start.dist = 10**models.start.dist
        
    models.pos.dist = 10**models.pos.dist
    models.lex.dist = 10**models.lex.dist
