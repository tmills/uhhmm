
cimport Indexer
cimport models
cimport numpy as np
cimport Sampler

cdef class HmmSampler(Sampler.Sampler):
    cdef Indexer.Indexer indexer
    cdef models.Models models
    cdef np.ndarray lexMatrix, lexMultiplier, dyn_prog, data, indices, indptr
    cdef int depth, maxLen

    cpdef public forward_pass(self, pi, list sent, int sent_index)
#    cdef _forward_sample_inner(self, pi, list sent, int g_max)
    cpdef reverse_sample(self, forward, pi, list sent, int sent_index)
    cdef _reverse_sample_inner(self, forward, pi, int sample_t, int t)
