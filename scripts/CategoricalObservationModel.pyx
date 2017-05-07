# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
import scipy.sparse
cimport Indexer
cimport numpy as np
cimport ObservationModel
import sys

cdef class CategoricalObservationModel(ObservationModel.ObservationModel):
    def __init__(self, indexer):
        self.indexer = indexer
        #ObvervationModel.ObservationModel.__init__(self)

    def set_models(self, models):
        self.indexer = Indexer.Indexer(models)
        g_len = models.pos.dist.shape[1]
        self.lexMatrix = np.matrix(models.lex.dist, copy=False)
        lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() // g_len)))
        print("Lexmultiplier has shape %s" % (str(lexMultiplier.shape)))
        self.data = lexMultiplier.data
        self.indices = lexMultiplier.indices
        self.indptr = lexMultiplier.indptr

    def get_probability_vector(self, token):
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes
        #print("Received token %d as input" % token)
        #sys.stdout.flush()
        lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
        #print("Shape of lex matrix is : ", self.lexMatrix.shape)
        #print("Multipyling lexMatrix with shape ?? by newly created lex multiplier of shape %s" % (str(lexMultiplier.shape)))
        #sys.stdout.flush()
        retVal = self.lexMatrix[:,token].transpose() * lexMultiplier
        #print("Matrix to be returned has shape %s" % (str(retVal.shape)))
        #sys.stdout.flush()
        return retVal
