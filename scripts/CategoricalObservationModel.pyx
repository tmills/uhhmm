#!/usr/bin/env python3

import numpy as np
import scipy.sparse

class CategoricalObservationModel:
    def __init__(self, indexer):
        self.indexer = indexer
        #ObvervationModel.ObservationModel.__init__(self)

    def set_models(self, models):
        self.lexMatrix = np.matrix(models.lex.dist, copy=False)
        self.lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() // g_len)))
        self.data = lexMultiplier.data
        self.indices = lexMultiplier.indices
        self.indptr = lexMultiplier.indptr

    def get_probability_vector(self):
        return self.lexMatrix[:,token].transpose() * self.lexMultiplier
