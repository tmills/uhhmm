# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
cimport Indexer
cimport numpy as np
cimport ObservationModel

cdef class CategoricalObservationModel(ObservationModel.ObservationModel):
    cdef Indexer.Indexer indexer
    cdef np.ndarray lexMatrix, data, indices, indptr
