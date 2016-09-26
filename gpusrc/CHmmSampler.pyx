# distutils: language = c++
# distutils: source = HmmSampler.cu
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
import State as PyState

## model class
cdef extern from "HmmSampler.h":
	cdef cppclass Model: # no nullary constructor
		Model(int, int, float*, int, int*, int, int*, int, float*, int, int, int, int, int, int, int) except +
		int get_depth()

cdef class GPUModel:
	cdef Model* c_model
	def __cinit__(self, models):
		(pi, lex_dist, maxes, depth) = models
		# int a_max, b_max, g_max
		(a_max, b_max, g_max) = maxes
		lex_num_rows = lex_dist.shape[0]
		lex_num_cols = lex_dist.shape[1]
		pi_num_rows = pi.shape[0]
		pi_num_cols = pi.shape[1]
		cdef np.ndarray[float, ndim=1, mode="c"] pi_data = pi.data
		cdef int pi_data_size = pi_data.size
		cdef np.ndarray[int, ndim=1, mode="c"] pi_indices = pi.indices
		cdef int pi_indices_size = pi_indices.size
		cdef np.ndarray[int, ndim=1, mode="c"] pi_indptr = pi.indptr
		cdef int pi_indptr_size = pi_indptr.size
		cdef np.ndarray[float, ndim=2, mode="c"] lex = lex_dist
		cdef int lex_dist_size = lex_dist.size
		self.c_model = new Model(pi_num_rows, pi_num_cols, &pi_data[0], pi_data_size, &pi_indptr[0], pi_indptr_size, &pi_indices[0], pi_indices_size,\
			&lex[0,0], lex_dist_size, lex_num_rows, lex_num_cols, a_max, b_max, g_max, depth)
	def __dealloc__(self):
		del self.c_model
	def get_depth(self):
		return self.c_model.get_depth()

## state class
cdef extern from "HmmSampler.h":
	cdef cppclass State:
		int depth
		int f
		int j
		vector[int] a
		vector[int] b
		int g

cdef wrap_state(State& c_state):
	cdef int depth = c_state.depth
	state = PyState.State(depth)
	state.f = c_state.f
	state.j = c_state.j
	state.a = np.array(c_state.a)
	state.b = np.array(c_state.b)
	state.g = c_state.g
	return state

# cdef class PyState:
# 	cdef State* c_state
# 	def __cinit__(self, State state):
# 		self.c_state = &state
# 	def get_states(self):
# 		return {'f' = self.c_state.f, 'j' = self.c_state.j, 'a' = self.c_state.a, 'b' = self.c_state.b, 'g' = self.c_state.g,\
# 		'depth' = self.c_state.depth}
## sampler class
cdef extern from "HmmSampler.h":
	cdef cppclass HmmSampler:
		HmmSampler() except +
		HmmSampler(int seed) except +
		void set_models(Model*)
		void initialize_dynprog(int)
		float forward_pass(vector[int], int)
		vector[State] reverse_sample(vector[int], int)


cdef class GPUHmmSampler:
	cdef HmmSampler hmmsampler
	def __cinit__(self, int seed = 0):
		if seed != 0:
			self.hmmsampler = HmmSampler(seed)
		else:
			self.hmmsampler = HmmSampler()
	def set_models(self, GPUModel model):
		self.hmmsampler.set_models(model.c_model)
	def initialize_dynprog(self, int k):
		self.hmmsampler.initialize_dynprog(k)
	def forward_pass(self, vector[int] sent, int sent_index):
		return self.hmmsampler.forward_pass(sent, sent_index)
	def reverse_sample(self, vector[int] sent, int sent_index):
		cdef vector[State] state_list = self.hmmsampler.reverse_sample(sent, sent_index)
		#states = self.hmmsampler.reverse_sample(sent, sent_index)
		#state_list = states[0]
		#print(states[1])
		wrapped_list = []
		for i in range(state_list.size()):
			wrapped_list.append(wrap_state(state_list[i]))
		return wrapped_list
	def sample(self, pi, vector[int] sent, int sent_index):  # need pi to conform to API
		# print "Sent Index is " + str(sent_index)
		log_probs = self.forward_pass(sent, sent_index) 
		states = self.reverse_sample(sent, sent_index) 
		return (states, log_probs)

# def test():
# 	import scipy.sparse
# 	lex = np.array([[1,2],[2,2.1]], dtype=np.float32)
# 	pi = scipy.sparse.csr_matrix((4, 5), dtype=np.float32)
# 	pi[2,2] = 1
# 	pi[3,4] = 1
# 	maxes = (3,2,3)
# 	a = PyModel(pi, lex, maxes)
# 	print a.get_depth()
