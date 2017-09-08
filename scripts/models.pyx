import numpy as np
cimport numpy as np
import logging
import distribution_sampler as sampler
from scipy.sparse import lil_matrix
import scipy.stats

## fullwiki_vecs_10dim.txt stats
#mean_mean = 0.18
#mean_stdev = 0.28
#stdev_mean = 0.4
#stdev_stdev = 0.04

## simple wiki stats:
mean_mean = 0.03
mean_stdev = 0.16
stdev_mean = 0.65
stdev_stdev = 0.07

# A mapping from input space to output space. The Model class can be
# used to store counts during inference, and then know how to resample themselves
# if given a base distribution.
# TODO: Sub-class for BooleanModel vs. InfiniteModel  with single sample()/resample() method
# and automatically adjusting sizes for infinite version.
cdef class Model:
    def __init__(self, corpus_shape=(1,1), name="Unspecified"):
        self.name = name
        self.corpus_shape = corpus_shape

## shape: shape of pos x tokens. shape[0]*shape[1] is the number of Gaussian
## parameters we need to store for the model.
## embeddings: the word embeddings for all the word types in this corpus
cdef class GaussianModel(Model):
    def __init__(self, shape, embeddings, corpus_shape=(1,1), name="Gaussian"):
        Model.__init__(self, corpus_shape, name)
        self.shape = shape
        self.embeddings = embeddings
        self.embed_dims = self.embeddings.shape[1]
        self.pairCounts = [] #np.zeros(shape, dtype=np.float32)
        self.globalPairCounts = np.zeros(shape, dtype=np.float32)
        #self.condCounts = [] #np.zeros(shape[0], dtype=np.int)
        ## Initialize distributions:
        self.dist = []

        mean_dist = scipy.stats.norm(mean_mean, mean_stdev)
        for pos_ind in range(shape[0]):
            self.pairCounts.append([])
            init_means = mean_dist.rvs(shape[1])
            stds = np.zeros(len(init_means)) + stdev_mean
            self.dist.append( scipy.stats.norm(init_means, stds) )
            #print("Mean vector for pos %d after initial sampling: " % (pos_ind), self.dist[-1].mean())

    def count(self, pos_ind, token_ind, val):
        ## we've seen a count of pos tag cond pos_ind and word index token_ind
        ## Go through embeddings matrix to create pair counts for each dimensions
        #self.condCounts[pos_ind] += val
        #for dim in range(self.embeddings.shape[1]):
        #    self.pairCounts[pos_ind, dim] += val * self.embeddings[token_ind][dim]
        if val != 1:
            raise Exception("For gaussian model, counts must be single increments")

        self.pairCounts[pos_ind].append(np.zeros(self.embed_dims) + self.embeddings[token_ind])


    def sampleGaussian(self):
#        if self.condCounts.sum() == 0:
#            ## This is called before we've done any sampling, and we already randomly
#            ## initialized in the constructor
#            return
        if len(self.pairCounts[0]) == 0:
            ## we haven't done any sampling yet
            return

        logging.info("Resampling gaussian observation models")

        sample_means = []
        sample_stdevs = []
        for pos_ind in range(len(self.pairCounts)-1):
            logging.info("Pair counts for POS%d with %d tokens assigned:" % (pos_ind, len(self.pairCounts[pos_ind])))
            #logging.info(self.pairCounts[pos_ind])
            stds = np.zeros(self.embed_dims) + stdev_mean
            if len(self.pairCounts[pos_ind]) > 0:
                #sample_means = self.pairCounts[pos_ind][:] / self.condCounts[pos_ind]
                sample_mean = np.mean(self.pairCounts[pos_ind], 0)
                sample_stdev = np.std(self.pairCounts[pos_ind], 0)
                sample_means.append(sample_mean)
                sample_stdevs.append(sample_stdev)
                self.dist[pos_ind] = scipy.stats.norm(sample_mean, stds)
                if pos_ind > 0:
                    logging.info("POS index=%d has distribution with mean %s and stdev %s" % (pos_ind, sample_mean, sample_stdev))

                ## FIXME: (see above)
    #            stds = np.zeros(len(sample_means)) + stdev_mean
    #            mean_sample_dist = scipy.stats.norm(sample_means, stds)
    #            if self.condCounts[pos_ind] == 0:
    #                logging.error("Error resampling Gaussian: One of the conditions (pos_ind=%d) had zero counts." % (pos_ind))
    #                ## don't assign it anything -- it keeps the same means as last sample.
    #            else:
    #                stds = np.zeros(self.pairCounts.shape[1]) + 0.4
    #                self.dist[pos_ind] = scipy.stats.norm(mean_sample_dist.rvs(self.pairCounts.shape[1]), stds)
            else:
                ## We didn't use this POS, so re-initialize it:
                mean_dist = scipy.stats.norm(mean_mean, mean_stdev)
                mean_init = mean_dist.rvs(self.embed_dims)
                self.dist[pos_ind] = scipy.stats.norm(mean_init, stds)
                if pos_ind > 0:
                    logging.info("POS index=%d has distribution with mean %s and stdev %s" % (pos_ind, sample_mean, sample_stdev))

            if np.isnan(self.dist[pos_ind].mean().max()):
                logging.error("Resampled gaussian and there is a nan in the mean vector")
                logging.error("self.dist[%d] = %s" %(pos_ind, str(self.dist[pos_ind].mean())))
                logging.error("sample_means: %s" % str(sample_means))

    def resetCounts(self):
        for pos_ind in range(len(self.pairCounts)):
            self.pairCounts[pos_ind] = []
            #self.condCounts[pos_ind] = 0
            #for dim in range(len(self.pairCounts)):
            #    self.pairCounts[pos_ind,dim] = 0
        self.pairCounts[0].append([1])

    def write_model(self, out_file, word_dict):
        f = open(out_file, 'w', encoding='utf-8')

        for pos_ind in range(len(self.pairCounts)):
            f.write("## writing pos_ind=%d\n" % pos_ind)
            for dim in range(self.embed_dims):
                f.write("## Writing pos_ind=%d, dim = %d\n"  % (pos_ind, dim))
                f.write("P( %d | POS%d ) ~ N(%f, %f)\n" % (dim, pos_ind, self.dist[pos_ind].mean()[dim], self.dist[pos_ind].std()[dim]))

        f.close()

    def __reduce__(self):
        d = {}
        d['shape'] = self.shape
        d['embeddings'] = self.embeddings
        d['pairCounts'] = self.pairCounts
        d['globalPairCounts'] = self.globalPairCounts
        #d['condCounts'] = self.condCounts
        d['dist'] = self.dist
        d['corpus_shape'] = self.corpus_shape
        d['name'] = self.name
        return (GaussianModel, (self.shape, self.embeddings, self.corpus_shape, self.name), d)

    def __setstate__(self, d):
        self.shape = d['shape']
        self.embeddings = d['embeddings']
        self.pairCounts = d['pairCounts']
        self.globalPairCounts = d['globalPairCounts']
        #self.condCounts = d['condCounts']
        self.dist = d['dist']
        self.corpus_shape = d['corpus_shape']
        self.name = d['name']


cdef class CategoricalModel(Model):
    def __init__(self, shape, float alpha=0.0, np.ndarray beta=None, corpus_shape=(1,1), name="Categorical"):
        Model.__init__(self, corpus_shape, name)
        self.shape = shape
        self.pairCounts = np.zeros(shape, dtype=np.int)
        self.globalPairCounts = np.zeros(shape, dtype=np.int)
        self.dist = np.random.random(shape)
        self.dist /= self.dist.sum(1, keepdims=True)
        self.dist = np.log10(self.dist)
        self.trans_prob = lil_matrix(self.corpus_shape)
        self.alpha = alpha
        if not beta is None:
            self.beta = beta
        else:
            self.beta = np.ones(shape[-1]) / shape[-1]

    def count(self, cond, out, val):
        out_counts = self.pairCounts[...,out]
        out_counts[cond] = out_counts[cond] + val
        if val < 0 and out_counts[cond] < 0:
            logging.error("Error! After a count there is a negative count")
            raise Exception

    def dec(self, cond, out):
        self.pairCounts[cond,out] -= 1

    def sampleDirichlet(self, base, L=1.0, from_global_counts=True):
        if from_global_counts:
            self.dist = np.log10((1-L)*10**self.dist + L*10**sampler.sampleDirichlet(self.globalPairCounts, base))
        else:
            self.dist = np.log10((1-L)*10**self.dist + L*10**sampler.sampleDirichlet(self.pairCounts, base))
#        print('Model name: %s' %self.name)
#        print(str(self.dist))
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
        m_copy = Model( self.shape, self.alpha, None if self.beta is None else self.beta.copy(), self.corpus_shape, self.name)
        m_copy.pairCounts = self.pairCounts.copy()
        m_copy.globalPairCounts = self.globalPairCounts.copy()
        m_copy.dist = self.dist.copy()
        return m_copy

    def __reduce__(self):
        #logging.info("Reduced called")
        d = {}
        d['shape'] = self.shape
        d['pairCounts'] = self.pairCounts
        d['globalPairCounts'] = self.globalPairCounts
        d['dist'] = self.dist
        d['trans_prob'] = self.trans_prob
        d['beta'] = self.beta
        d['corpus_shape'] = self.corpus_shape
        d['alpha'] = self.alpha
        d['name'] = self.name

        return (CategoricalModel, (self.shape, self.alpha, self.beta, self.corpus_shape, self.name), d)

    def __setstate__(self, d):
        #logging.info("Into set_state")
        self.shape = d['shape']
        self.pairCounts = d['pairCounts']
        self.globalPairCounts = d['globalPairCounts']
        self.dist = d['dist']
        self.trans_prob = d['trans_prob']
        self.beta = d['beta']
        self.corpus_shape = d['corpus_shape']
        self.alpha = d['alpha']
        self.name = d['name']

    # Assumes beta stick has already been broken
    def add_outcome(self):
        new_ix = self.shape[-1]
        self.pairCounts = np.insert(self.pairCounts, new_ix, 1, -1)
        self.globalPairCounts = np.insert(self.globalPairCounts, new_ix, 0, -1)
        self.dist = np.insert(self.dist, new_ix, 0, -1)
        new_shape = list(self.shape)
        new_shape[-1] += 1
        self.shape = tuple(new_shape)

    def remove_outcome(self, outcome):
        self.pairCounts = np.delete(self.pairCounts, outcome, -1)
        self.globalPairCounts = np.delete(self.globalPairCounts, outcome, -1)
        self.dist = np.delete(self.dist, outcome, -1)
        new_shape = list(self.shape)
        new_shape[-1] -= 1
        self.shape = tuple(new_shape)

        # Capture lost beta
        lost_beta = self.beta[outcome]
        # Remove outcome from beta
        self.beta = np.delete(self.beta, outcome, -1)
        # Uniformly redistribute lost beta
        self.beta += lost_beta/self.shape[-1]

    def add_condition(self, axis=-2):
        new_shape = self.shape
        new_ix = self.shape[axis]
        self.pairCounts = np.insert(self.pairCounts, new_ix, 0, axis)
        self.globalPairCounts = np.insert(self.globalPairCounts, new_ix, 0, axis)
        self.dist = np.insert(self.dist, new_ix, 0, axis)
        new_shape = list(self.shape)
        new_shape[axis] += 1
        self.shape = tuple(new_shape)

    def remove_condition(self, cond, axis=-2):
        self.pairCounts = np.delete(self.pairCounts, cond, axis)
        self.globalPairCounts = np.delete(self.globalPairCounts, cond, axis)
        self.dist = np.delete(self.dist, cond, axis)
        new_shape = list(self.shape)
        new_shape[axis] -= 1
        self.shape = tuple(new_shape)

    def break_beta_stick(self, gamma):
        beta_end = self.beta[-1]
        self.beta = np.append(self.beta, np.zeros(1))
        old_portion = new_portion = 0

        while old_portion == 0 or new_portion == 0:
            old_group_fraction = np.random.beta(1, gamma)
            old_portion = old_group_fraction * beta_end
            new_portion = (1-old_group_fraction) * beta_end

        self.beta[-2] = old_portion
        self.beta[-1] = new_portion

cdef class Models:
    def __init__(self):
        self.models = []

    def initialize_as_fjabp(self, max_output, params, corpus_shape, depth, a_max, b_max, g_max, lex=None):

        ## FJ model:
        self.fj = [None] * depth
        ## Active self:
        self.act = [None] * depth
        self.root = [None] * depth
        ## Reduce self:
        self.cont = [None] * depth
        self.exp = [None] * depth
        self.next = [None] * depth
        self.start = [None] * depth

        for d in range(0, depth):
            ## One fork model:
            self.fj[d] = CategoricalModel((a_max, b_max, b_max, g_max, 4), alpha=float(params.get('alphafj')), name="FJ"+str(d))

            ## One active model:
            self.act[d] = CategoricalModel((a_max, b_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape, name="A|00_"+str(d))

            self.root[d] = CategoricalModel((b_max, g_max, a_max), alpha=float(params.get('alphaa')), corpus_shape=corpus_shape, name="A|10_"+str(d))

            ## four awaited self:
            self.cont[d] = CategoricalModel((b_max, g_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|11_"+str(d))

            self.exp[d] = CategoricalModel((g_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|10_"+str(d))

            self.next[d] = CategoricalModel((a_max, b_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|01_"+str(d))

            self.start[d] = CategoricalModel((a_max, a_max, b_max), alpha=float(params.get('alphab')), corpus_shape=corpus_shape, name="B|00_"+str(d))


        ## one pos model:
        self.pos = CategoricalModel((b_max, g_max), alpha=float(params.get('alphag')), corpus_shape=corpus_shape, name="POS")

        if lex is None:
            logging.info("Initializing default lexical model as Categorical")
            ## one lex model (categorical default)
            self.lex = CategoricalModel((g_max, max_output+1), alpha=float(params.get('alphah')), name="Lex")
        else:
            logging.info("Using passed in lexical model")
            ## Let the calling entity worry about the lex model.
            self.lex = lex


        self.append(self.fj)
        self.append(self.act)
        self.append(self.root)
        self.append(self.cont)
        self.append(self.start)
        self.append(self.exp)
        self.append(self.next)
        self.append(self.pos)
        self.append(self.lex)

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

    def break_a_stick(self, gamma):
        depth = len(self.fj)
        self.root[0].break_beta_stick(gamma)

        for d in range(depth):
            self.fj[0].add_condition(0)

            self.root[d].add_outcome()
            self.root[d].beta[:] = self.root[0].beta[:]
            self.act[d].add_outcome()
            self.act[d].beta[:] = self.root[0].beta[:]

            self.exp[d].add_condition(1)
            self.next[d].add_condition(1)
            self.start[d].add_condition(0)
            self.start[d].add_condition(1)

    def break_b_stick(self, gamma):
        depth = len(self.fj)
        self.cont[0].break_beta_stick(gamma)

        for d in range(depth):
            self.fj[d].add_condition(1)
            self.fj[d].add_condition(2)

            self.root[d].add_condition(0)
            self.act[d].add_condition(1)

            self.cont[d].add_outcome()
            self.cont[d].beta[:] = self.cont[0].beta[:]
            self.next[d].add_outcome()
            self.cond[d].beta[:] = self.cont[0].beta[:]

        self.pos.add_condition(0)

    def break_g_stick(self, gamma):
        depth = len(self.fj)
        self.pos.break_beta_stick(gamma)

        for d in range(depth):
            self.fj[d].add_condition(3)
            self.root[d].add_condition(1)
            self.cont[d].add_condition(1)
            self.exp[d].add_condition(0)

        self.pos.add_outcome()
        self.lex.add_condition(0)

    def remove_pos(self, pos):
        depth = len(self.fj)
        for d in range(depth):
            self.fj[d].remove_condition(pos, 3)
            self.root[d].remove_condition(pos, 1)
            self.cont[d].remove_condition(pos, 1)
            self.exp[d].remove_condition(pos, 0)

        self.pos.remove_outcome(pos)
        self.lex.remove_condition(pos, 0)

    ## Init is used if there are no counts yet -- buffers the pseudocounts to avoid underflow
    def resample_all(self, decay=1.0, from_global_counts=False, init=False):
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        depth = len(self.fj)
        fj_base = self.fj[0].alpha * self.fj[0].beta + int(init)
        a_base =  self.root[0].alpha * self.root[0].beta + int(init)
        b_base = self.cont[0].alpha * self.cont[0].beta + int(init)
        g_base = self.pos.alpha * self.pos.beta + int(init)

        for d in range(depth-1, -1, -1):
            self.fj[d].sampleDirichlet(fj_base if d == 0 else fj_base + self.fj[d-1].pairCounts * self.fj[d].alpha, decay, from_global_counts)
            self.start[d].sampleDirichlet(b_base if d == 0 else b_base + self.start[d-1].pairCounts * self.start[d].alpha, decay, from_global_counts)
            self.exp[d].sampleDirichlet(b_base if d == 0 else b_base + self.exp[d-1].pairCounts * self.exp[d].alpha, decay, from_global_counts)
            self.cont[d].sampleDirichlet(b_base if d == 0 else b_base + self.cont[d-1].pairCounts * self.cont[d].alpha, decay, from_global_counts)
            self.next[d].sampleDirichlet(b_base if d == 0 else b_base + self.next[d-1].pairCounts * self.next[d].alpha, decay, from_global_counts)
            self.act[d].sampleDirichlet(a_base if d == 0 else a_base + self.act[d-1].pairCounts * self.act[d].alpha, decay, from_global_counts)
            self.root[d].sampleDirichlet(a_base if d == 0 else a_base + self.root[d-1].pairCounts * self.root[d].alpha, decay, from_global_counts)

        # Resample pos and make sure only the null awa can generate the null tag
        self.pos.sampleDirichlet(g_base, decay, from_global_counts)
        #self.lex.dist[1:,0].fill(-np.inf)


        # Resample lex and make sure the null tag can only generate the null word
        if type(self.lex).__name__ == 'CategoricalModel':
            h_base = self.lex.alpha * self.lex.beta + int(init)
            self.lex.sampleDirichlet(h_base, decay, from_global_counts)
            self.lex.dist[0,0] = 0.0
            self.lex.dist[0,1:].fill(-np.inf)
        elif type(self.lex).__name__ == 'GaussianModel':
            self.lex.sampleGaussian()
        else:
            logging.error("The type of the lexical model is not understood: %s" % type(self.lex).__name__)

    def increment_global_counts(self):
        depth = len(self.fj)
        for d in range(depth-1, -1, -1):
            self.fj[d].globalPairCounts += self.fj[d].pairCounts
            self.start[d].globalPairCounts += self.start[d].pairCounts
            self.exp[d].globalPairCounts += self.exp[d].pairCounts
            self.cont[d].globalPairCounts += self.cont[d].pairCounts
            self.next[d].globalPairCounts += self.next[d].pairCounts
            self.act[d].globalPairCounts += self.act[d].pairCounts
            self.root[d].globalPairCounts += self.root[d].pairCounts
        self.pos.globalPairCounts += self.pos.pairCounts
        #self.lex.globalPairCounts += self.lex.pairCounts
