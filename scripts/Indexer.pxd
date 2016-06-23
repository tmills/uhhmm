
cimport numpy as np
import numpy as np

cdef class Indexer:
    cdef int depth, fj_size, a_max, b_max, g_max, state_size, a_size, b_size
    
    cpdef public int extractPos(self, int index)
    
    cpdef public extractStacks(self, int index)
    
    cpdef int getStateIndex(self, int f, int j, np.ndarray a, np.ndarray b, int g)
    
    cpdef getStateTuple(self, int f, int j, np.ndarray a, np.ndarray b)