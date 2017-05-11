# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

import models
cimport models

cdef class ObservationModel:

    cdef set_models(self, models.Model)
    cpdef get_probability_vector(self, token)
