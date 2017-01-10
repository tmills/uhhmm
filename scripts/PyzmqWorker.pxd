
cimport Indexer
from cpython cimport bool

cdef class PyzmqWorker:
    cdef int jobs_port, results_port, models_port, maxLen, out_freq, tid, seed, debug_level, batch_size
    cdef bool quit, gpu
    cdef Indexer.Indexer indexer
    cdef str host
    cdef tuple model_file_sig
    cdef float scheduled_time_of_death, longest_wait_for_new_model, longest_wait_processing_sentences, longest_wait_processing_rows
