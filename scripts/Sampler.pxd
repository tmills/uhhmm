

cdef class Sampler:
    cdef public float ff_time
    cdef public float bs_time

    cpdef public sample(self, pi, list sent, int sent_index)