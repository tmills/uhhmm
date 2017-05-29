import nltk
import numpy as np
import logging
from pcfg_translator import normalize_a_tensor

class PCFG_model:
    def __init__(self, abp_domain_size, len_vocab):
        self.nonterms = [nltk.grammar.Nonterminal(str(x)) for x in range(abp_domain_size + 1)]
        self.indices_keys, self.keys_indices = self.build_pcfg_indices(abp_domain_size, len_vocab)
        self.size = len(self.indices_keys)
        self.alpha = 1.0
        self.init_counts()

    def __getitem__(self, item):
        assert isinstance(item, tuple), "PCFG can only look up tuples"
        try:
            if isinstance(item[1], int):
                return self.indices_keys[item[0]][item[1]]
            elif isinstance(item[1], tuple):
                return self.keys_indices[item[0]][item[1]]
        except:
            print(item)
            print(type(item[1][0]))
            print(self.keys_indices)
            raise Exception("Blah")

    def __len__(self):
        return self.size

    def init_counts(self):
        self.counts = {x:np.zeros(len(self.indices_keys[x])) + self.alpha for x in self.indices_keys}

    def set_alpha(self, alpha):
        self.alpha = alpha

    def sample(self, pcfg_counts, annealing_coeff=1.0, normalize = False): # used as the normal sampling procedure
        self._reset_counts()
        self._update_counts(pcfg_counts)
        sampled_pcfg = self._sample_model(annealing_coeff, normalize = normalize)
        sampled_pcfg = self._translate_model_to_pcfg(sampled_pcfg)
        return sampled_pcfg

    def _reset_counts(self):
        for parent in self.counts:
            self.counts[parent].fill(self.alpha)

    def _update_counts(self, pcfg_counts):
        for parent in pcfg_counts:
            for children in pcfg_counts[parent]:
                index = self[(parent, children)]
                self.counts[parent][index] += pcfg_counts[parent][children]

    def _sample_model(self, annealing_coeff=1.0, normalize=False):
        logging.info("resample the pcfg model with alpha {} and annealing coeff {}.".format(self.alpha, annealing_coeff))
        dists = {x:np.random.dirichlet(self.counts[x]) ** annealing_coeff for x in self.counts}
        if normalize:
            dists = {x:normalize_a_tensor(dists[x]) for x in dists}
        return dists

    def _translate_model_to_pcfg(self, dists):
        pcfg = {x:{} for x in dists}
        for parent in pcfg:
            for index, value in enumerate(dists[parent]):
                rhs = self[(parent, index)]
                pcfg[parent][rhs] = value
        return pcfg


    def build_pcfg_indices(self, abp_domain_size, len_vocab):
        keys_indices = {x:{} for x in self.nonterms}
        indices_keys = {x:[] for x in self.nonterms}
        for parent in range(abp_domain_size+1): # from 0 - domain_size
            if parent == 0:
                for child in range(abp_domain_size+1): # from 0 - domain_size
                    if child != 0:
                        rule = nltk.grammar.Production(self.nonterms[parent], (self.nonterms[child], self.nonterms[parent]))
                        keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                        indices_keys[rule.lhs()].append(rule.rhs())
                else:
                    rule = nltk.grammar.Production(self.nonterms[parent], ('-ROOT-',))
                    keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                    indices_keys[rule.lhs()].append(rule.rhs())
            else:

                for l_child in range(1, abp_domain_size + 1):
                    for r_child in range(1, abp_domain_size + 1):
                        rule = nltk.grammar.Production(self.nonterms[parent], (self.nonterms[l_child], self.nonterms[r_child]))
                        keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                        indices_keys[rule.lhs()].append(rule.rhs())
                for lex in range(1, len_vocab+1):
                    rule = nltk.grammar.Production(self.nonterms[parent], (str(lex),))
                    keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                    indices_keys[rule.lhs()].append(rule.rhs())

        return indices_keys, keys_indices

if __name__ == '__main__':
    abp_domain_size = 5
    vocab_size = 3
    pcfg_model = PCFG_model(5, 3)
    pcfg_counts_model = {}
    pcfg_counts_model[nltk.grammar.Nonterminal('1')] = {}
    pcfg_counts_model[nltk.grammar.Nonterminal('1')][(nltk.grammar.Nonterminal('1'), nltk.grammar.Nonterminal('3'))] = 100
    pcfg_model._update_counts(pcfg_counts_model)
    print(pcfg_model.counts)
    pcfg_model._reset_counts()
    print(pcfg_model.counts)
    pcfg_model._update_counts(pcfg_counts_model)
    pcfg_model_dict = pcfg_model.sample(pcfg_counts_model)
    print(pcfg_model.sample(pcfg_counts_model)[nltk.grammar.Nonterminal('1')][(nltk.grammar.Nonterminal('1'), nltk.grammar.Nonterminal('3'))])
    print(pcfg_model_dict[nltk.grammar.Nonterminal('0')])
