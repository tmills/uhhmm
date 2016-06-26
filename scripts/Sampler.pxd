

cdef class Sampler:
    cdef public float ff_time, bs_time

    cpdef public sample(self, pi, list sent, int sent_index)