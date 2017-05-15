import numpy as np
cimport numpy as np
import logging
import distribution_sampler as sampler
from scipy.sparse import lil_matrix

# A mapping from input space to output space. The Model class can be
# used to store counts during inference, and then know how to resample themselves
# if given a base distribution.
# TODO: Sub-class for BooleanModel vs. InfiniteModel  with single sample()/resample() method
# and automatically adjusting sizes for infinite version.
cdef class Model:

    def __init__(self, shape, float alpha=0.0, beta=None, corpus_shape=(1,1), name="Unspecified"):
        ## Initialize with ones to prevent underflow during distribution sampling
        ## at iteration 0
        self.shape = shape
        self.corpus_shape = corpus_shape
        self.pairCounts = np.zeros(shape, dtype=np.float)
        self.dist = np.random.random(shape)
        self.dist /= self.dist.sum(1, keepdims=True)
        self.dist = np.log10(self.dist)
        self.trans_prob = lil_matrix(corpus_shape)
        self.alpha = alpha
        if beta is not None:
            self.beta = beta
        else:
            self.beta = np.ones(shape[-1]) / shape[-1]
        self.name = name

    def count(self, cond, out, val):
        out_counts = self.pairCounts[...,out]
        out_counts[cond] = out_counts[cond] + val
        if val < 0 and out_counts[cond] < 0:
            logging.error("Error! After a count there is a negative count")
            raise Exception

    def dec(self, cond, out):
        self.pairCounts[cond,out] -= 1

    def sampleDirichlet(self, base):
        self.dist = sampler.sampleDirichlet(self.pairCounts, base)
#        print('Model name: %s' %self.name)
#        print('Count sums: %s' %self.pairCounts.sum(axis=(0,1)))
#        with np.errstate(divide='ignore', invalid='ignore'):
#            norm_cts = np.nan_to_num(self.pairCounts/self.pairCounts.sum(2)[:,:,None]).sum(axis=(0,1))
#            print('Normalized count sums: %s' %(norm_cts / norm_cts.sum()))
#            print('Sampled distribution sums: %s' %((10**self.dist).sum(axis=(0,1))))
#            print('Sampled marginal distribution: %s' %((10**self.dist).sum(axis=(0,1))/(10**self.dist).sum()))
#            print('')

    def resetCounts(self):
        self.pairCounts[:] = 0

    def copy(self):
        m_copy = Model( (self.pairCounts.shape[0], self.pairCounts.shape[1]), self.alpha, None if self.beta is None else self.beta.copy(), (self.trans_prob.shape[0], self.trans_prob.shape[1]))
        m_copy.pairCounts = self.pairCounts.copy()
        m_copy.dist = self.dist.copy()
        return m_copy
        
    def __reduce__(self):
        #logging.info("Reduced called")
        d = {}
        d['pairCounts'] = self.pairCounts
        d['dist'] = self.dist
        d['trans_prob'] = self.trans_prob
        d['beta'] = self.beta
        d['alpha'] = self.alpha
        d['name'] = self.name
        d['corpus_shape'] = self.corpus_shape
        d['shape'] = self.shape

        return (Model, (self.shape, self.alpha, self.beta, self.corpus_shape, self.name), d)
        
    def __setstate__(self, d):
        #logging.info("Into set_state")
        self.shape = d['shape']
        self.pairCounts = d['pairCounts']
        self.dist = d['dist']
        self.trans_prob = d['trans_prob']
        self.beta = d['beta']
        self.alpha = d['alpha']
        self.name = d['name']
        self.corpus_shape = d['corpus_shape']
        
cdef class Models:
    def __init__(self):
        self.models = []

#    def resample_all(self):
#        for model in self.models:
#            model.dist = sampleDirichlet(model)

    def resetAll(self):
        for model in self.models:
            if type(model) == list:
                for submodel in model:
                    submodel.resetCounts()
            else:
                model.resetCounts()

    def append(self, model):
        self.models.append(model)

    def __reduce__(self):
        d = {}
        d['F'] = self.F
        d['J'] = self.J
        d['A'] = self.A
        d['B_J1'] = self.B_J1
        d['B_J0'] = self.B_J0
        d['pos'] = self.pos
        d['lex'] = self.lex
        return (Models, (), d)
    
    def __setstate__(self, d):
        self.F = d['F']
        self.J = d['J']
        self.A = d['A']
        self.B_J1 = d['B_J1']
        self.B_J0 = d['B_J0']
        self.pos = d['pos']
        self.lex = d['lex']
        self.models = [self.F, self.J, self.A, self.B_J1, self.B_J0, self.pos, self.lex]

    def __iter__(self):
        for index, model in enumerate(['F', 'J', 'A', 'B_J1', 'B_J0', 'pos', 'lex']):
            yield model, getattr(self, model)