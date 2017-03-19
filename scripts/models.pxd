
import numpy as np
cimport numpy as np

cdef class Model:
    cdef public np.ndarray pairCounts, dist, u, beta
    cdef public float alpha
    cdef public str name
    cdef public trans_prob
    
cdef class Models:
    cdef list models
    cdef public list F, J, A, B_J0, B_J1
    cdef public Model pos, lex

