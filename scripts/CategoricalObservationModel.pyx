# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
import scipy.sparse
cimport Indexer
cimport numpy as np
cimport PosDependentObservationModel
import sys
cimport models

cdef class CategoricalObservationModel(PosDependentObservationModel.PosDependentObservationModel):

    cdef get_pos_probability_vector(self, token):
        return self.lexMatrix[:,token]
