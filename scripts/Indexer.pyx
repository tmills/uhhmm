# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
cimport cython
import State
import numpy as np
cimport numpy as np
#from Sampler import *

cdef class Indexer:
    cdef int depth, depth_size, a_max, b_max, g_max, state_size, a_size, b_size
    def __init__(self, models):
        self.depth = len(models.fork)
        self.depth_size = 1 + self.depth * 4
        self.a_max = models.act[0].dist.shape[-1]
        self.b_max = models.start[0].dist.shape[-1]
        self.g_max = models.pos.dist.shape[-1]
        self.state_size = self.depth_size * ((self.a_max * self.b_max) ** self.depth) * self.g_max 
        self.a_size = self.a_max**self.depth
        self.b_size = self.b_max**self.depth
    
    def getVariableMaxes(self):
        return (self.a_max, self.b_max, self.g_max)

    def get_state_size(self):   
        return self.state_size
    
    #@profile
    def extractState(self, int index):
        (fStack, jStack, aStack, bStack, g) = self.extractStacks(index)
        state = State.State(self.depth)
        state.f = fStack
        state.j = jStack
        state.a = aStack
        state.b = bStack
        state.g = g
        return state

    cpdef public int extractPos(self, int index):
        cdef int fj_ind, a_ind, b_ind, g, max_d, f_val, j_val
        cdef np.ndarray f, j, a, b
        
        (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (self.depth_size, self.a_size, self.b_size, self.g_max))
        return g

    #@profile
    cpdef public extractStacks(self, int index):
        cdef int fj_ind, a_ind, b_ind, g, max_d, f_val, j_val
        cdef np.ndarray f, j, a, b
        
        (fj_ind, a_ind, b_ind, g) = np.unravel_index(index, (self.depth_size, self.a_size, self.b_size, self.g_max))
    
        f = np.zeros(self.depth, dtype=int) - 1 
        j = np.zeros(self.depth, dtype=int) - 1
    
        ## if fj_ind == 0 that is the case where f[d] = j[d] = -1 for all d (which is what 
        ## we initialized f and j to so we don't need to handle that case)
        if fj_ind > 0:
            (max_d, f_val, j_val) = np.unravel_index(fj_ind-1, (self.depth, 2, 2))
            f[max_d] = f_val
            j[max_d] = j_val
    
        a = np.array(np.unravel_index(a_ind, [self.a_max] * self.depth), dtype=int)
        b = np.array(np.unravel_index(b_ind, [self.b_max] * self.depth), dtype=int)
    
        return f, j, a, b, g

    ## We compose a state index from separate fj, a, b, and g indexes.
    ## Things get a little trickier at d == D because at time t=1 we technically have
    ## an expansion at d=-1, meaning the fj stack is completely empty. (At d=1 we just
    ## fudged it and had the expansion at d=0. The result is we have to check whether the
    ## depth is -1 and if so we have a special case, since there's only one place where it
    ## would happen and we don't want to blow up the state space.
    cpdef int getStateIndex(self, np.ndarray f, np.ndarray j, np.ndarray a, np.ndarray b, int g):
        cdef int d, cur_depth, fj_stack, index
        
        for d in range(0, self.depth):
            if f[d] >= 0:
                cur_depth = d
                break
            cur_depth = -1
        
        if cur_depth == -1:
            fj_stack = 0
        else:
            fj_stack = 1 + np.ravel_multi_index((cur_depth, f[cur_depth], j[cur_depth]), (self.depth, 2, 2))

        a_stack  = np.ravel_multi_index(a, [self.a_max] * self.depth)
        b_stack = np.ravel_multi_index(b, [self.b_max] * self.depth)
        
        index = np.ravel_multi_index((fj_stack, a_stack, b_stack, g), (self.depth_size, self.a_size, self.b_size, self.g_max))
    
        return index
    
    cpdef getStateTuple(self, np.ndarray f, np.ndarray j, np.ndarray a, np.ndarray b):
        cdef int start
        start = self.getStateIndex(f,j,a,b,0)
        return (start,start+self.g_max-1)
