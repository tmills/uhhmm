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
cdef class Model

cdef class Model:

    def __init__(self, shape, float alpha=0.0, np.ndarray beta=None, corpus_shape=(0,0), name="Unspecified"):
        ## Initialize with ones to prevent underflow during distribution sampling
        ## at iteration 0
        self.pairCounts = np.ones(shape, dtype=np.int)
        self.dist = np.random.random(shape)
        self.dist /= self.dist.sum(1, keepdims=True)
        self.dist = np.log10(self.dist)
        self.u = np.array([])
        self.trans_prob = lil_matrix(corpus_shape)
        self.alpha = alpha
        self.beta = beta
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
        m_copy = Model( (self.pairCounts.shape[0], self.pairCounts.shape[1]), self.alpha, None if self.beta == None else self.beta.copy(), (self.trans_prob.shape[0], self.trans_prob.shape[1]))
        m_copy.pairCounts = self.pairCounts.copy()
        m_copy.dist = self.dist.copy()
        return m_copy
        
    def __reduce__(self):
        #logging.info("Reduced called")
        d = {}
        d['pairCounts'] = self.pairCounts
        d['dist'] = self.dist
        d['u'] = self.u
        d['trans_prob'] = self.trans_prob
        d['beta'] = self.beta
        d['alpha'] = self.alpha
        d['name'] = self.name
        
        return (Model, ( (self.pairCounts.shape[0], self.pairCounts.shape[1]),), d)
        
    def __setstate__(self, d):
        #logging.info("Into set_state")
        self.pairCounts = d['pairCounts']
        self.dist = d['dist']
        self.u = d['u']
        self.trans_prob = d['trans_prob']
        self.beta = d['beta']
        self.alpha = d['alpha']
        self.name = d['name']        
        
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
        d['fj'] = self.fj
        d['act'] = self.act
        d['root'] = self.root
        d['next'] = self.next
        d['cont'] = self.cont
        d['exp'] = self.exp
        d['start'] = self.start
        d['pos'] = self.pos
        d['lex'] = self.lex
        return (Models, (), d)
    
    def __setstate__(self, d):
        self.fj = d['fj']
        self.act = d['act']
        self.root = d['root']
        self.next = d['next']
        self.cont = d['cont']
        self.exp = d['exp']
        self.start = d['start']
        self.pos = d['pos']
        self.lex = d['lex']
        self.models = [self.fj, self.act, self.root, self.next, self.cont, self.exp, self.start, self.pos, self.lex]

