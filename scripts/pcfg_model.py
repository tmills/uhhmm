import logging
import os.path
import gzip
import nltk
import numpy as np
from scipy.stats import dirichlet
import collections


def normalize_a_tensor(tensor):
    return tensor / (
    np.sum(tensor, axis=-1, keepdims=True) + 1e-20)  # to supress zero division warning


def calculate_alpha(num_terms, num_sents, non_root_nonterm_mask, num_term_types,
                    num_nonterminal_types, scale):
    assert scale <= 1, "the scale hyperparameter for beta cannot be larger than 1!"
    logging.info("num of sents : {}; num of tokens: {}; num of types : {}; num of nonterm types: {"
                 "}".format(num_sents, num_terms, num_term_types, num_nonterminal_types))
    num_terminal_nodes = num_terms
    num_nonterminal_nodes = num_terms - num_sents
    num_nonterminal_rules = np.sum(non_root_nonterm_mask)
    total_non_term_pseudo_counts = scale * num_nonterminal_nodes
    total_term_pseudo_counts = scale * num_terminal_nodes
    avg_non_term_pseudo_counts = total_non_term_pseudo_counts / num_nonterminal_rules / num_nonterminal_types
    avg_term_pseudo_counts = total_term_pseudo_counts / num_term_types / num_nonterminal_types
    assert avg_term_pseudo_counts <= 1, "the average terminal pseudo count is larger than 1! please tune down th" \
                                        "e scale hyperparameter!"
    non_term_betas = non_root_nonterm_mask * avg_non_term_pseudo_counts
    term_betas = (non_root_nonterm_mask == 0).astype(float) * avg_term_pseudo_counts
    beta = term_betas + non_term_betas
    logging.info('the average non-terminal pseudo count is {:.2f}, and terminal {:.2f}. '
                 'scale is {}'.format(
        avg_non_term_pseudo_counts
        , avg_term_pseudo_counts, scale))
    return beta, avg_non_term_pseudo_counts, avg_term_pseudo_counts


class PCFG_model:
    def __init__(self, abp_domain_size, len_vocab, num_sents, num_words, log_dir='.', iter=0,
                 word_dict_file=None, autocorr_lags=(50,100)):
        self.autocorr_lags = autocorr_lags
        self.prev_models = collections.deque([], max(self.autocorr_lags))
        self.iter_autocorrs = []
        self.abp_domain_size = abp_domain_size
        self.len_vocab = len_vocab
        self.num_sents = num_sents
        self.num_words = num_words
        self.non_root_nonterm_mask = []
        self.nonterms = [nltk.grammar.Nonterminal(str(x)) for x in range(abp_domain_size + 1)]
        self.indices_keys, self.keys_indices = self.build_pcfg_indices(abp_domain_size, len_vocab)
        self.size = len(self.indices_keys)
        self.alpha_range = []
        self.alpha = 0
        self.alpha_scale = 0
        self.nonterm_alpha, self.term_alpha = 0, 0
        self.alpha_array_flag = False
        self.counts = {}
        self.right_branching_tendency = 0.0
        # self.init_counts()
        self.unannealed_dists = {}
        self.log_dir = log_dir
        self.log_mode = 'w'
        self.iter = iter
        self.nonterm_log_path = os.path.join(log_dir, 'pcfg_nonterms.gzip')
        self.term_log_path = os.path.join(log_dir, 'pcfg_terms.gzip')
        self.hypparams_log_path = os.path.join(log_dir, 'pcfg_hypparams.txt')
        self.counts_log_path = os.path.join(log_dir, 'pcfg_counts_info.txt')
        self.word_dict = self._read_word_dict_file(word_dict_file)
        self.log_probs = 0
        self.annealed_counts = {}

    def set_log_mode(self, mode):
        self.log_mode = mode  # decides whether append to log or restart log

    def start_logging(self):
        self.nonterm_log = gzip.open(self.nonterm_log_path, self.log_mode+'t')
        self.term_log = gzip.open(self.term_log_path, self.log_mode+'t')
        self.counts_log = open(self.counts_log_path, self.log_mode)
        if self.log_mode == 'w':
            non_term_header = ['iter', ]
            term_header = ['iter', ]
            for lhs in self.nonterms:
                for rhs in self.keys_indices[lhs]:
                    if len(rhs) == 1:
                        if rhs[0] == '-ROOT-':
                            term_header.append(str(lhs) + '->' + rhs[0])
                        else:
                            term_header.append(str(lhs) + '->' + str(self.word_dict[int(rhs[0])]))
                    else:
                        non_term_header.append(str(lhs) + '->' + str(rhs))
            self.nonterm_log.write('\t'.join(non_term_header) + '\n')
            self.term_log.write('\t'.join(term_header) + '\n')
        self.hypparam_log = open(self.hypparams_log_path, self.log_mode)
        nonterms_counts_header = [str(x) for x in self.nonterms if x.symbol() != '0']
        nonterms_non_counts_header = [x+'_non' for x in nonterms_counts_header]
        if self.log_mode == 'w':
            self.hypparam_log.write('iter\tlogprob\talpha\tac\tRB\n')
            self.counts_log.write('iter\t{}\t{}\n'.format('\t'.join(
                nonterms_counts_header), '\t'.join(nonterms_non_counts_header)))


    def _log_dists(self, dists):
        non_term_header = [self.iter, ]
        term_header = [self.iter, ]
        for lhs in self.nonterms:
            for rhs in self.keys_indices[lhs]:
                if len(rhs) == 1:
                    index = self[(lhs, rhs)]
                    term_header.append(dists[lhs][index])
                else:
                    index = self[(lhs, rhs)]
                    non_term_header.append(dists[lhs][index])
        self.nonterm_log.write('\t'.join([str(x) for x in non_term_header]) + '\n')
        self.term_log.write('\t'.join([str(x) for x in term_header]) + '\n')

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
        if isinstance(self.alpha, np.ndarray):
            # print(list(self.indices_keys.keys()))
            # print(list(map(len, self.indices_keys.values())))

            for x in self.indices_keys:
                if x.symbol() == '0':
                    self.counts[x] = np.ones(len(self.indices_keys[x]))
                else:
                    self.counts[x] = np.zeros(len(self.indices_keys[x])) + self.alpha
        elif isinstance(self.alpha, dict):
            print(self.alpha)
            for x in self.indices_keys:
                if x.symbol() == '0':
                    self.counts[x] = np.ones(len(self.indices_keys[x]))
                else:
                    self.counts[x] = np.zeros(len(self.indices_keys[x])) + self.alpha[x]
        else:
            self.counts = {x: np.zeros(len(self.indices_keys[x])) + self.alpha
                           for x in self.indices_keys}

    def set_alpha(self, alpha_range, alpha=0.0, alpha_scale=0.0, alpha_array_flag=False):
        self.alpha_array_flag = alpha_array_flag
        # print(alpha, alpha_scale, alpha_array_flag, self.counts)
        if isinstance(alpha_range, list):
            self.alpha_range = alpha_range
        else:
            self.alpha_range = [float(x) for x in alpha_range.split(',')]
        if alpha != 0 and alpha_scale != 0:
            logging.warning("Do NOT set init_alpha and alpha_scale at the same time."
                            "init_alpha will be ignored.")
        if alpha_scale != 0:
            self.alpha_scale = alpha_scale
            self.alpha, self.nonterm_alpha, self.term_alpha = \
                calculate_alpha(self.num_words, self.num_sents, self.non_root_nonterm_mask,
                                self.len_vocab, self.abp_domain_size, alpha_scale)
            alpha_vec = np.zeros_like(self.non_root_nonterm_mask, dtype=np.float)
            for ele_id, ele in enumerate(self.non_root_nonterm_mask):
                if ele == 1:
                    alpha_vec[ele_id] = self.nonterm_alpha
                elif ele == 0:
                    alpha_vec[ele_id] = self.term_alpha
            self.alpha = alpha_vec
        elif alpha != 0.0 and not self.alpha_array_flag:
            assert alpha >= alpha_range[0] and alpha <= alpha_range[1]
            self.alpha, self.nonterm_alpha, self.term_alpha = alpha, alpha, alpha
        elif alpha != 0.0 and self.alpha_array_flag:
            self.alpha = self.nonterm_alpha = self.term_alpha = {x:alpha for x in self.indices_keys}
        else:
            self.alpha = sum(alpha_range) / len(alpha_range)
        self.init_counts()

    def sample(self, pcfg_counts, annealing_coeff=1.0, normalize=False,
               sample_alpha_flag=False, right_branching_tendency=0.0):  # used as the normal
        # sampling procedure
        self.right_branching_tendency = right_branching_tendency
        if sample_alpha_flag or self.alpha_array_flag:
            self._sample_alpha()
        self._reset_counts()
        self._update_counts(pcfg_counts)
        sampled_pcfg = self._sample_model(annealing_coeff, normalize=normalize)
        # self._calc_autocorr()
        sampled_pcfg = self._translate_model_to_pcfg(sampled_pcfg)
        self.nonterm_log.flush()
        self.term_log.flush()
        self.hypparam_log.flush()
        self.counts_log.flush()
        return sampled_pcfg

    def _reset_counts(self):
        for parent in self.counts:
            if isinstance(self.alpha, np.ndarray):
                if parent.symbol() == '0':
                    self.counts[parent].fill(1)
                else:
                    self.counts[parent] = np.copy(self.alpha)
            elif isinstance(self.alpha, dict):
                self.counts[parent].fill(self.alpha[parent])
            else:
                self.counts[parent].fill(self.alpha)

    def _update_counts(self, pcfg_counts):
        self.nonterm_total_counts = {}
        self.nonterm_non_total_counts = {}
        for parent in pcfg_counts:
            if parent.symbol() != '0':
                self.nonterm_total_counts[parent] = sum(pcfg_counts[parent].values())
                self.nonterm_non_total_counts[parent] = 0
            for children in pcfg_counts[parent]:
                index = self[(parent, children)]
                self.counts[parent][index] += pcfg_counts[parent][children]
                if len(children) < 2 and parent.symbol() != '0':
                    self.nonterm_non_total_counts[parent] += pcfg_counts[parent][children]

    # def _calc_autocorr(self):
    #     for lag in self.autocorr_lags:
    #         if len(self.prev_models) < lag:
    #             self.iter_autocorrs.append(np.inf)
    #         else:


    def _sample_alpha(self, step_size=0.01):  # sampling the hyperparamter for the dirichlets
        if not self.unannealed_dists:
            pass
        elif self.alpha_scale != 0.:
            logging.warning('Sampling different alphas for nonterms and terms are not supported!')
        else:
            if not self.alpha_array_flag:
                old_f_val = 0.0
                new_f_val = 0.0
                for parent in self.unannealed_dists.keys():
                    if parent.symbol() == '0':
                        continue
                    dist = self.unannealed_dists[parent]
                    alpha_vec = np.zeros_like(dist)
                    alpha_vec.fill(self.alpha)
                    old_f_val += dirichlet.logpdf(dist, alpha_vec)
                new_alpha = 0
                while new_alpha <= self.alpha_range[0] or new_alpha > self.alpha_range[1] or \
                        new_alpha == 0:
                    new_alpha = self.alpha + np.random.normal(0.0, step_size)
                for parent in self.unannealed_dists.keys():
                    if parent.symbol() == '0':
                        continue
                    alpha_vec = np.zeros_like(dist)
                    alpha_vec.fill(new_alpha)
                    new_f_val += dirichlet.logpdf(dist, alpha_vec)
                acceptance_thres = np.log(np.random.uniform(0.0, 1.0))
                mh_ratio = new_f_val - old_f_val
                if mh_ratio > acceptance_thres:
                    self.alpha = new_alpha
                    self.term_alpha = new_alpha
                    self.nonterm_alpha = new_alpha
                    logging.info(
                        'pcfg alpha samples a new value {} with log ratio {}/{}'.format(new_alpha,
                                                                                        mh_ratio,
                                                                                        acceptance_thres))
            else:
                old_f_val = 0.0
                new_f_val = 0.0
                for parent in self.unannealed_dists.keys():
                    if parent.symbol() == '0':
                        continue
                    dist = self.unannealed_dists[parent]
                    old_alpha_vec = np.zeros_like(dist)
                    old_alpha_vec.fill(self.alpha[parent])
                    old_f_val = dirichlet.logpdf(dist, old_alpha_vec)
                    new_temp_alpha = 0
                    while new_temp_alpha <= self.alpha_range[0] or new_temp_alpha > self.alpha_range[1] or \
                                    new_temp_alpha == 0:
                        new_temp_alpha = self.alpha[parent] + np.random.normal(0.0, step_size)

                    new_alpha_vec = np.zeros_like(dist)
                    new_alpha_vec.fill(new_temp_alpha)
                    new_f_val = dirichlet.logpdf(dist, new_alpha_vec)
                    acceptance_thres = np.log(np.random.uniform(0.0, 1.0))
                    mh_ratio = new_f_val - old_f_val
                    if mh_ratio > acceptance_thres:
                        self.alpha[parent] = new_temp_alpha

                        logging.info(
                            'pcfg alpha samples a new value {} for {} with log ratio {}/{}'.format(
                                new_temp_alpha, parent, mh_ratio, acceptance_thres))
                self.term_alpha = self.alpha
                self.nonterm_alpha = self.alpha

    def _sample_model(self, annealing_coeff=1.0, normalize=False):
        logging.info("resample the pcfg model with nonterm alpha {}, term alpha {} and annealing "
                     "coeff {}.".format(self.nonterm_alpha, self.term_alpha, annealing_coeff))
        if self.log_probs != 0:
            # self.hypparam_log.write('\t'.join([str(x) for x in [self.iter, self.log_probs,
            #                                                     (self.nonterm_alpha,
            #                                                      self.term_alpha),
            #                                                     annealing_coeff, self.right_branching_tendency]]) + '\n')

            # self.counts_log.write('\t'.join([self.iter,] + [str(nonterm_total_counts[x]) for x in
            #                       self.nonterms if x.symbol() != '0']) + '\n')
            self.hypparam_log.write('\t'.join([str(x) for x in [self.iter, self.log_probs,
                                                                (self.nonterm_alpha,
                                                                 self.term_alpha),
                                                                annealing_coeff,
                                                                self.right_branching_tendency]]))
            self.counts_log.write('\t'.join([str(self.iter),] + [str(self.nonterm_total_counts[p])
                                                             for p in
                                  self.nonterms if str(p) != '0'] + [str(
                self.nonterm_non_total_counts[x]) for x in self.nonterms if str(x) != '0']) +
                                  '\n')

        dists = {}
        if annealing_coeff != 1.0:
            if not self.alpha_array_flag:
                self.anneal_counts = {
                    x: (self.counts[x] - self.alpha) * annealing_coeff if x.symbol()
                                                                      != '0' else
                    (self.counts[x] - 1) * annealing_coeff
                    for x in self.counts}
                self.unannealed_dists = {x: np.random.dirichlet(self.anneal_counts[x] + self.alpha)
                    if x.symbol() != "0" else
                    np.random.dirichlet(self.anneal_counts[x] + 1)
                                         for x in self.anneal_counts}
            else:
                self.anneal_counts = {
                    x: (self.counts[x] - self.alpha[x]) * annealing_coeff if x.symbol()
                                                                             != '0' else
                    (self.counts[x] - 1) * annealing_coeff
                    for x in self.counts}
                self.unannealed_dists = {x: np.random.dirichlet(self.anneal_counts[x] +
                                                                self.alpha[x])
                    if x.symbol() != "0" else
                    np.random.dirichlet(self.anneal_counts[x] + 1)
                                             for x in self.anneal_counts}
            # if normalize:
            #     dists = {x: normalize_a_tensor(dists[x]) for x in dists}
        else:
            self.anneal_counts = self.counts
            self.unannealed_dists = {x: np.random.dirichlet(self.counts[x]) for x in self.counts}
        dists = self.unannealed_dists
        for dist in dists.values():
            assert np.all(dist != 0), "there are 0s in distributions from Dirichlet! It is " \
                                      "usually because alpha is too small."
        # print(dists)
        # print(np.sum(dists[nltk.grammar.Nonterminal('1')], axis=0))
        self._log_dists(dists)
        # print(dists)
        # print(list(map(len, dists.values())))
        return dists

    def _translate_model_to_pcfg(self, dists):
        pcfg = {x: {} for x in dists}
        for parent in pcfg:
            # print(parent)
            for index, value in enumerate(dists[parent]):
                rhs = self[(parent, index)]
                pcfg[parent][rhs] = value
        return pcfg

    def build_pcfg_indices(self, abp_domain_size, len_vocab):
        keys_indices = {x: {} for x in self.nonterms}
        indices_keys = {x: [] for x in self.nonterms}
        for parent in range(abp_domain_size + 1):  # from 0 - domain_size
            if parent == 0:
                for child in range(abp_domain_size + 1):  # from 0 - domain_size
                    if child != 0:
                        rule = nltk.grammar.Production(self.nonterms[parent],
                                                       (
                                                       self.nonterms[child], self.nonterms[parent]))
                        keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                        indices_keys[rule.lhs()].append(rule.rhs())
                else:
                    rule = nltk.grammar.Production(self.nonterms[parent], ('-ROOT-',))
                    keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                    indices_keys[rule.lhs()].append(rule.rhs())
            else:

                for l_child in range(1, abp_domain_size + 1):
                    for r_child in range(1, abp_domain_size + 1):
                        rule = nltk.grammar.Production(self.nonterms[parent],
                                                       (self.nonterms[l_child],
                                                        self.nonterms[r_child]))
                        keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                        indices_keys[rule.lhs()].append(rule.rhs())
                        if parent == 1:
                            self.non_root_nonterm_mask.append(1)
                for lex in range(1, len_vocab + 1):
                    rule = nltk.grammar.Production(self.nonterms[parent], (str(lex),))
                    keys_indices[rule.lhs()][rule.rhs()] = len(keys_indices[rule.lhs()])
                    indices_keys[rule.lhs()].append(rule.rhs())
                    if parent == 1:
                        self.non_root_nonterm_mask.append(0)
        self.non_root_nonterm_mask = np.array(self.non_root_nonterm_mask)
        print(self.non_root_nonterm_mask)
        return indices_keys, keys_indices

    def _read_word_dict_file(self, word_dict_file):
        f = open(word_dict_file, 'r', encoding='utf-8')
        word_dict = {}
        for line in f:
            (word, index) = line.rstrip().split(" ")
            word_dict[int(index)] = word
        return word_dict


if __name__ == '__main__':
    abp_domain_size = 5
    vocab_size = 3
    pcfg_model = PCFG_model(5, 3)
    pcfg_counts_model = {}
    pcfg_counts_model[nltk.grammar.Nonterminal('1')] = {}
    pcfg_counts_model[nltk.grammar.Nonterminal('1')][
        (nltk.grammar.Nonterminal('1'), nltk.grammar.Nonterminal('3'))] = 100
    pcfg_model._update_counts(pcfg_counts_model)
    print(pcfg_model.counts)
    pcfg_model._reset_counts()
    print(pcfg_model.counts)
    pcfg_model._update_counts(pcfg_counts_model)
    pcfg_model_dict = pcfg_model.sample(pcfg_counts_model, annealing_coeff=0.5, normalize=True)
    print(
        pcfg_model_dict[nltk.grammar.Nonterminal('1')][
            (nltk.grammar.Nonterminal('1'), nltk.grammar.Nonterminal('3'))])
    print(pcfg_model_dict[nltk.grammar.Nonterminal('1')])
