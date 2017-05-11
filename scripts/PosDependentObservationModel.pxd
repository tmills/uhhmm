# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport models

cdef class PossDependentObservationModel:
    cdef get_probability_vector(self, models.Model):
