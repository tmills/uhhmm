import numpy as np
cimport numpy as np
import logging
import distribution_sampler as sampler
from scipy.sparse import lil_matrix
import scipy.stats
from scipy.stats import invgamma
from scipy.stats import gamma

## fullwiki_vecs_10dim.txt stats
#mean_mean = 0.18
#mean_stdev = 0.28
#stdev_mean = 0.4
#stdev_stdev = 0.04

## simple wiki stats:
mean_mu = 0.0
mean_sigmasq = 0.16

prec_alpha = 4.0
prec_beta = 4.0

cdef invgammapdf(x, alpha, beta):
    scaled_alpha = x / beta
    return invgamma(x, alpha) / beta

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
    def __init__(self, shape, embeddings, corpus_shape=(1,1), name="Gaussian", centroids=None):
        Model.__init__(self, corpus_shape, name)
        self.shape = shape
        self.embeddings = embeddings
        self.embed_dims = self.embeddings.shape[1]
        self.pairCounts = [] #np.zeros(shape, dtype=np.float32)
        self.globalPairCounts = np.zeros(shape, dtype=np.float32)
        #self.condCounts = [] #np.zeros(shape[0], dtype=np.int)
        ## Initialize distributions:
        self.dist = []
        self.centroids = centroids

        mean_dist = scipy.stats.norm(mean_mu, mean_sigmasq)
        # logging.info("Initializing pos distributions with count %d,%d" % (shape[0], shape[1]))
        for pos_ind in range(shape[0]):
            self.pairCounts.append([])
            if pos_ind == 0 or pos_ind == (shape[0] - 1) or self.centroids is None:
                # logging.info("Initializing pos ind=%d randomly." % (pos_ind))
                init_means = mean_dist.rvs(shape[1])
            else:
                # logging.info("Initializing pos ind=%d from centroid index %d" % (pos_ind, pos_ind-1))
                init_means = self.centroids[pos_ind-1]

            ## Initialize all equally: (means the first iteration will essentailly randomly sample words
            ## from each distribution)
            init_means = np.zeros(shape[1])
            stds = np.zeros(len(init_means)) + 4.0 # FIXME with sampling?
            self.dist.append( scipy.stats.norm(init_means, stds) )

            # logging.info("Mean vector for pos %d after initial sampling: %s" % (pos_ind+1, self.dist[-1].mean()))

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
        if len(self.pairCounts[0]) == 0:
            ## we haven't done any sampling yet
            return

        logging.info("Resampling gaussian observation models")

        sample_means = []
        sample_stdevs = []
        for pos_ind in range(len(self.pairCounts)-1):
            logging.info("Recalculating distribution for POS%d with %d tokens assigned" % (pos_ind, len(self.pairCounts[pos_ind])))
            # logging.info(self.pairCounts[pos_ind])
            # stds = np.zeros(self.embed_dims) + stdev_mean
            num_samples = len(self.pairCounts[pos_ind])
            if num_samples > 1:
                #sample_means = self.pairCounts[pos_ind][:] / self.condCounts[pos_ind]
                sample_mean = np.mean(self.pairCounts[pos_ind], 0)
                sample_stdev = np.std(self.pairCounts[pos_ind], 0)
                sample_var = sample_stdev**2.
                if np.any(sample_var == 0):
                    logging.warn("Sample variance is 0 with %d samples so there may be a divide by zero error!" % (num_samples))
                    #raise Exception
                ## precision = 1 / variance
                sample_precision = 1. / (sample_var)
                mean_dist_arg1 = (mean_sigmasq / ((sample_var / num_samples) + mean_sigmasq)) * sample_mean # + (sample_var / ((sample_var/num_samples) + mean_sigmasq)) * mean_mean_prior   ## Last part is commented out -- as long as mean_mean_prior is 0 that term will be 0
                mean_dist_arg2 = 1. / (1./mean_sigmasq + num_samples / sample_var)
                # logging.info("New means being drawn from distribution with mean %s and stdevs %s with %d samples" % (mean_dist_arg1, mean_dist_arg2, num_samples))
                mean_dist = scipy.stats.norm(mean_dist_arg1, mean_dist_arg2)

                prec_dist_arg1 = prec_alpha + num_samples / 2
                internal = np.array(((self.pairCounts[pos_ind] - self.dist[pos_ind].mean())**2).sum(0))
                prec_dist_arg2 = 1 / (prec_beta + 0.5 * internal)
                # logging.info("Arguments to gamma distribution are %s and %s" % (str(prec_dist_arg1), str(prec_dist_arg2)))
                prec_samples = gamma.rvs(prec_dist_arg1, scale=prec_dist_arg2)
                if np.any(prec_samples == 0):
                    logging.warn("Precision samples is zero so there may be a divide by zero error!")
                var_samples = 1 / prec_samples
                stdev_samples = np.sqrt(var_samples)
                self.dist[pos_ind] = scipy.stats.norm(mean_dist.rvs(), stdev_samples)
                # if pos_ind > 0:
                #     logging.info("POS index=%d has new distribution with mean %s and stdev %s" % (pos_ind, sample_mean, sample_stdev))

            else:
                ## We didn't use this POS, so re-initialize it:
                ## FIXME - sample variance the same way we do in the constructor
                mean_dist = scipy.stats.norm(mean_mu, mean_sigmasq)
                mean_init = mean_dist.rvs(self.embed_dims)
                self.dist[pos_ind] = scipy.stats.norm(mean_init, 2.0)
                # if pos_ind > 0:
                    # logging.info("POS index=%d has distribution with mean %s and stdev %s" % (pos_ind, sample_mean, sample_stdev))

            if np.isnan(self.dist[pos_ind].mean().max()):
                logging.error("Resampled gaussian and there is a nan in the mean vector")
                logging.error("self.dist[%d] = %s" %(pos_ind, str(self.dist[pos_ind].mean())))
                logging.error("sample_means: %s" % str(sample_means))
                
        self.resetCounts()
        logging.info("Done resampling gaussian word distributions.")

    def resetCounts(self):
        logging.info("Reset counts called in gaussian model.")
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
        d['centroids'] = self.centroids
        return (GaussianModel, (self.shape, self.embeddings, self.corpus_shape, self.name, self.centroids), d)

    def __setstate__(self, d):
        self.shape = d['shape']
        self.embeddings = d['embeddings']
        self.pairCounts = d['pairCounts']
        self.globalPairCounts = d['globalPairCounts']
        #self.condCounts = d['condCounts']
        self.dist = d['dist']
        self.corpus_shape = d['corpus_shape']
        self.name = d['name']
        self.centroids = d['centroids']

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
        m_copy = CategoricalModel( self.shape, self.alpha, None if self.beta is None else self.beta.copy(), self.corpus_shape, self.name)
        m_copy.pairCounts = self.pairCounts.copy()
        m_copy.globalPairCounts = self.globalPairCounts.copy()
        m_copy.dist = self.dist.copy()
        return m_copy

    def write_model(self, out_file, word_dict):
        return

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
