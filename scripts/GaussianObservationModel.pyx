# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
cimport Indexer
cimport numpy as np
cimport PosDependentObservationModel
cimport models

cdef class GaussianObservationModel(PosDependentObservationModel.PosDependentObservationModel):
    cdef set_models(self, uhhmm_models):
        PosDependentObservationModel.PosDependentObservationModel.set_models(self, uhhmm_models)
        self.lex = uhhmm_models.lex

    ## In this version of the observation model, token is a continuous vector.
    ## Here we compute the joint probability of the observation dimensions given
    ## the different POS tags. So for each tag, we compute the product of
    ## probabilities of output of each dimension given the parameters to the
    ## gaussian distribution for that POS tag and dimension.
    ## For multinomial, we have G * W params.
    ## For gaussian we have G x 2 x D params for D-dimensional embeddings.
    ## Assume that the values inside the models.lex array are scipy.stats.norm distributions.
    cdef get_pos_probability_vector(self, token):
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes
        token_vec = self.lex.embeddings[token]
        
        retVec = [0.0]
        for g in range(1,g_max-1):
            prob = 1.0
            for ind in range(len(token_vec)):
                dim_prob = self.lex.dist[g][ind].pdf(token_vec[ind])
                prob *= dim_prob
                #print("POS: %d, ind=%d, vec_val=%f, dim_prob=%f, and prob=%.8f" % (g, ind, token_vec[ind], dim_prob, prob))
                #print("PDF for this pos and ind has mean %f" % self.lex.dist[g][ind].mean())
            
            if prob <= 0.0:
                print("Total probability underflow (= 0.0)")
            retVec.append(prob)

        retVec.append(0.0)
        return np.array(retVec)
