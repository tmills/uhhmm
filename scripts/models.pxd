
import numpy as np
import scipy.sparse.lil_matrix as lil_matrix
cimport numpy as np

cdef class Model:
    cdef public np.ndarray pairCounts, dist, u, beta
    cdef public float alpha
    cdef public str name
    
cdef class Models:
    cdef list models
    cdef public list fork, trans, reduce, act, root, cont, start, exp, next
    cdef public Model pos, lex

