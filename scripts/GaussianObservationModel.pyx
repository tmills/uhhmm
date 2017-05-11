# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
cimport Indexer
cimport numpy as np
cimport ObservationModel

cdef class GaussianObservationModel(ObservationModel.ObservationModel):
    def __init__(self, indexer):
        self.indexer = indexer

    def set_models(self, models):
        self.indexer = Indexer.Indexer(models)
        g_len = models.pos.dist.shape[1]

    ## In this version of the observation model, token is a continuous vector.
    ## Here we compute the joint probability of the observation dimensions given
    ## the different POS tags. So for each tag, we compute the product of
    ## probabilities of output of each dimension given the parameters to the
    ## gaussian distribution for that POS tag and dimension.
    ## For multinomial, we have G * W params.
    ## For gaussian we have G x 2 x D params for D-dimensional embeddings.
    ## Assume that the values inside the models.lex array are scipy.stats.norm distributions.
    def get_probability_vector(self, token):
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes

        retVec = [0.0]
        for g in range(1,g_max):
            prob = 1.0
            for ind in range(len(token)):
                prob *= self.models.lex[g].pdf(token[ind])
            retVec.append(prob)

        retVec.append(0.0)
        return retVec
