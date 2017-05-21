
import numpy as np
cimport numpy as np

cdef class Model:
    cdef public tuple corpus_shape
    cdef public str name

cdef class Models:
    cdef list models
    cdef public list fj, act, root, cont, start, exp, next
    cdef public Model pos
    cdef public lex

cdef class CategoricalModel(Model):
    cdef public tuple shape
    cdef public np.ndarray pairCounts, globalPairCounts, dist, u, beta
    cdef public float alpha
    cdef public trans_prob

cdef class GaussianModel(Model):
    pass
