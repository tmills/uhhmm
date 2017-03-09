
import numpy as np
cimport numpy as np

cdef class Model:
    cdef public tuple shape, corpus_shape
    cdef public np.ndarray pairCounts, globalPairCounts, dist, u, beta
    cdef public float alpha
    cdef public str name
    cdef public trans_prob
    
cdef class Models:
    cdef list models
    cdef public list fj, act, root, cont, start, exp, next
    cdef public Model pos, lex

