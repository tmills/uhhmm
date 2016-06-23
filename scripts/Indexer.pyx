# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
cimport cython
import State
import numpy as np
cimport numpy as np

cdef class Indexer:
    #cdef int depth, fj_size, a_max, b_max, g_max, state_size, a_size, b_size
    def __init__(self, models):
        self.depth = len(models.fork)
        self.fj_size = 4
        self.a_max = models.act[0].dist.shape[-1]
        self.b_max = models.start[0].dist.shape[-1]
        self.g_max = models.pos.dist.shape[-1]
        self.state_size = self.fj_size * ((self.a_max * self.b_max) ** self.depth) * self.g_max 
        self.a_size = self.a_max**self.depth
        self.b_size = self.b_max**self.depth
    
    def getVariableMaxes(self):
        return (self.a_max, self.b_max, self.g_max)

    def get_state_size(self):   
        return self.state_size
    
    #@profile
    def extractState(self, int index):
        (f, j, aStack, bStack, g) = self.extractStacks(index)
        state = State.State(self.depth)
        state.f = f
        state.j = j
        state.a = aStack
        state.b = bStack
        state.g = g
        return state

    cpdef public int extractPos(self, int index):
        cdef int fj_ind, a_ind, b_ind, g, max_d, f_val, j_val
        
        (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (self.fj_size, self.a_size, self.b_size, self.g_max))
        return g

    #@profile
    cpdef public extractStacks(self, int index):
        cdef int fj_ind, a_ind, b_ind, g, max_d, f_val, j_val, f, j
        cdef np.ndarray a, b
        
        (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (self.fj_size, self.a_size, self.b_size, self.g_max))
        
        f = 0 if fj_ind / 2 == 0 else 1
        j = 0 if fj_ind % 2 == 0 else 1
           
        a = np.array(np.unravel_index(a_ind, [self.a_max] * self.depth), dtype=int)
        b = np.array(np.unravel_index(b_ind, [self.b_max] * self.depth), dtype=int)
    
        return f, j, a, b, g

    ## We compose a state index from separate fj, a, b, and g indexes.
    ## Things get a little trickier at d == D because at time t=1 we technically have
    ## an expansion at d=-1, meaning the fj stack is completely empty. (At d=1 we just
    ## fudged it and had the expansion at d=0. The result is we have to check whether the
    ## depth is -1 and if so we have a special case, since there's only one place where it
    ## would happen and we don't want to blow up the state space.
    cpdef int getStateIndex(self, int f, int j, np.ndarray a, np.ndarray b, int g):
        cdef int d, fj_ind, index
        
        fj_ind = 0
        if f == 1:
            fj_ind = 2
        
        if j == 1:
            fj_ind += 1
            
        a_stack  = np.ravel_multi_index(a, [self.a_max] * self.depth)
        b_stack = np.ravel_multi_index(b, [self.b_max] * self.depth)
        
        index = np.ravel_multi_index((fj_ind, a_stack, b_stack, g), (self.fj_size, self.a_size, self.b_size, self.g_max))
    
        return index
    
    cpdef getStateTuple(self, int f, int j, np.ndarray a, np.ndarray b):
        cdef int start
        start = self.getStateIndex(f,j,a,b,0)
        return (start,start+self.g_max-1)

