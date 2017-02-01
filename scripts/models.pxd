
import numpy as np
cimport numpy as np

cpdef class Model:
    cdef public np.ndarray pairCounts, dist, u, beta
    cdef public float alpha
    cdef public str name
    
cdef class Models:
    cdef list models
    cdef public list fork, trans, reduce, act, root, cont, start, exp, next
    cdef public Model pos, lex

