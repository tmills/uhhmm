from left_corner2normal_tree_converter import full_chain_convert
import nltk
import numpy as np
from copy import deepcopy
"""
this file is for translating sequences of states to pcfg counts and back to uhhmm counts
the main function is translate_through_pcfg
"""
RB_TREES = ["-::ACT0/AWA0::+::POS1::1 -::ACT2/AWA2::+::POS1::1 +::ACT1/AWA2::-::POS2::2",
            "-::ACT0/AWA0::+::POS1::1 -::ACT2/AWA2::-::POS2::2"]

def translate_through_pcfg(seqs_of_states, depth, abp_domain_size):
    trees = []
    for seq in seqs_of_states:
        tree = full_chain_convert(seq, depth)
        trees.append(tree)
    count_dict = extract_counts(trees, abp_domain_size)
    return count_dict

def extract_counts(trees, abp_domain_size):
    pcfg = {}
    top_cat = nltk.grammar.Nonterminal('0')
    zero_lexical = nltk.grammar.Production(top_cat, ('-ROOT-',))
    for tree in trees:
        # rules = _extract_counts_single_tree(tree, nonterms)
        top_node = tree.label()
        top_rule = nltk.grammar.Production(top_cat, (nltk.grammar.Nonterminal(top_node), top_cat))
        pcfg_rules = tree.productions() + [top_rule, zero_lexical]
        for rule in pcfg_rules:
            if rule.lhs() not in pcfg:
                pcfg[rule.lhs()] = {}
            pcfg[rule.lhs()][rule.rhs()] = pcfg[rule.lhs()].get(rule.rhs(), 0) + 1
    pcfg_counts = deepcopy(pcfg)
    for lhs in pcfg:
        total = sum(pcfg[lhs].values())
        for rhs in pcfg[lhs]:
            pcfg[lhs][rhs] /= total
    return pcfg, pcfg_counts

def _build_nonterminals(abp_domain_size):
    # we build the 0 nonterminal just for convenience. it is not used.
    return nltk.grammar.nonterminals(','.join([str(x) for x in range(0, abp_domain_size+1)])+',...' )

def _calc_delta(pcfg, J, abp_domain_size, d, nonterminals):
    delta = np.zeros((2, J, abp_domain_size + 1, d))  # the delta model. s * d * i
    for a_index in range(abp_domain_size + 1):
        # if a_index == 0:
        #     continue
        a = nonterminals[a_index]
        if a in pcfg:
            lexical_sum = sum([items[1] for items in pcfg[a].items() if len(items[0]) == 1 and
                           nltk.grammar.is_terminal(items[0][0])])
            delta[0, 1:, a_index, :] = lexical_sum
            delta[1, 1:, a_index, :] = lexical_sum
    for i_index in range(2, J):
        for a_index in range(abp_domain_size+1):  # the category labels correspond to their true labels. there is no 0
            if a_index == 0:
                continue
            a = nonterminals[a_index]
            for depth in range(d):
                nonterm_sum_a = 0
                nonterm_sum_b = 0
                if a in pcfg:
                    for rhs in pcfg[a]:
                        if len(rhs) == 1:
                            continue
                        prob = pcfg[a][rhs]
                        a_prime = int(rhs[0].symbol())
                        b_prime = int(rhs[1].symbol())
                        nonterm_sum_a += prob * delta[0, i_index-1, a_prime, depth] * delta[1, i_index-1, b_prime, depth]
                        if depth + 1 < d:
                            nonterm_sum_b += prob * delta[0, i_index-1, a_prime, depth + 1] * delta[1, i_index-1, b_prime, depth]
                    delta[0, i_index, a_index, depth] += nonterm_sum_a
                    delta[1, i_index, a_index, depth] += nonterm_sum_b
    # for i in range(2):
    #     for j in range(J):
    #          print(i, j, delta[i, j])
    return delta[0, -1,...].T, delta[1,-1,...].T

def _calc_gamma(deltas, pcfg, pcfg_counts, d):
    delta_A, delta_B = deltas
    gamma_As, gamma_Bs = [], []
    gamma_A_counts, gamma_B_counts = [], []
    for depth in range(d):
        gamma_As.append({})
        gamma_Bs.append({})
        gamma_A_counts.append({})
        gamma_B_counts.append({})
        for lhs in pcfg:
            for rhs in pcfg[lhs]:
                if any([nltk.grammar.is_terminal(x) for x in rhs]):
                    continue
                if lhs not in gamma_As[depth]:
                    gamma_As[depth][lhs] = {}
                    gamma_Bs[depth][lhs] = {}
                    gamma_A_counts[depth][lhs] = {}
                    gamma_B_counts[depth][lhs] = {}
                if rhs not in gamma_As[depth][lhs]:
                    gamma_As[depth][lhs][rhs] = 0
                    gamma_Bs[depth][lhs][rhs] = 0
                    gamma_A_counts[depth][lhs][rhs] = 0
                    gamma_B_counts[depth][lhs][rhs] = 0
                gamma_As[depth][lhs][rhs] = pcfg[lhs][rhs] * delta_A[depth][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())] / delta_A[depth][int(lhs.symbol())]
                gamma_A_counts[depth][lhs][rhs] = pcfg_counts[lhs][rhs] * delta_A[depth][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())] / delta_A[depth][int(lhs.symbol())]
                if depth + 1 < d:
                    gamma_Bs[depth][lhs][rhs] = pcfg[lhs][rhs] * delta_A[depth+1][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())]  / delta_B[depth][int(lhs.symbol())]
                    gamma_B_counts[depth][lhs][rhs] = pcfg_counts[lhs][rhs] * delta_A[depth+1][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())]  / delta_B[depth][int(lhs.symbol())]
    return gamma_As, gamma_Bs, gamma_A_counts, gamma_B_counts

def _calc_expected_counts(gammas, pcfg_counts, J, d, abp_domain_size):
    gamma_As, gamma_Bs = gammas
    gamma_star_plus = np.zeros((J, d, abp_domain_size+1, abp_domain_size+1))
    preterm_marginal_distr = np.zeros(abp_domain_size+1)
    for lhs in pcfg_counts:  # counting the accumulative probabilities for any preterminals
        for rhs in pcfg_counts[lhs]:
            if all([nltk.grammar.is_terminal(x) for x in rhs]):
                lhs_index = int(lhs.symbol())
                preterm_marginal_distr[lhs_index] += pcfg_counts[lhs][rhs]
    for j in range(J):
        for depth in range(d):
            for lhs in gamma_As[depth]:
                lhs_index = int(lhs.symbol())
                for rhs in gamma_As[depth][lhs]:
                    a_prime = int(rhs[0].symbol())
                    if j == 0:
                        gamma_star_plus[j, depth, lhs_index, a_prime] += gamma_Bs[depth][lhs][rhs]
                    else:
                        gamma_star_plus[j, depth, :-1, a_prime] += gamma_star_plus[j-1, depth, :-1, lhs_index] \
                                                                 * gamma_As[depth][lhs][rhs]
    # print(gamma_star_plus)
    gamma_star_plus_final = np.sum(gamma_star_plus, axis=0)
    return gamma_star_plus_final, preterm_marginal_distr

def _calc_f_model_bp(gamma_stars, d, abp_domain_size, normalize=False):
    gamma_star_plus, preterm_marginal_distr = gamma_stars
    f_model = np.zeros((d, abp_domain_size+2, abp_domain_size+2, 2))
    for depth in range(d):
        for lhs in range(0, abp_domain_size+1):
            for rhs in range(0, abp_domain_size+1):
                if lhs != rhs:
                    f_model[depth, lhs, rhs,1] = gamma_star_plus[depth, lhs, rhs] * preterm_marginal_distr[lhs]
                else:
                    f_model[depth, lhs, rhs, 0] = preterm_marginal_distr[lhs]
                    f_model[depth, lhs, rhs, 1] = gamma_star_plus[depth, lhs, rhs] * preterm_marginal_distr[lhs]
        if depth == 0:
            f_model[0, 0, 0, 0] = 0
    if normalize:
        return _normalize_a_tensor(f_model)
    return f_model

def _calc_f_model(gamma_stars, d, abp_domain_size, normalize=False):
    gamma_star_plus, preterm_marginal_distr = gamma_stars
    f_model = np.zeros((d, abp_domain_size+2, 2))
    for depth in range(d):
        for lhs in range(0, abp_domain_size+1):
            for rhs in range(0, abp_domain_size+1):
                if lhs != rhs:
                    f_model[depth, lhs, 1] += gamma_star_plus[depth, lhs, rhs] * preterm_marginal_distr[lhs]
                else:
                    f_model[depth, lhs, 0] += preterm_marginal_distr[lhs]
                    f_model[depth, lhs, 1] += gamma_star_plus[depth, lhs, rhs] * preterm_marginal_distr[lhs]
        if depth == 0:
            f_model[0, 0, 0] = 0
    if normalize:
        return _normalize_a_tensor(f_model)
    return f_model

def _calc_j_model(gamma_counts, gamma_stars, d, abp_domain_size, normalize=False):
    gamma_star_plus, preterm_marginal_distr = gamma_stars
    gamma_A_counts, gamma_B_counts = gamma_counts
    j_model = np.zeros((d, abp_domain_size+2, abp_domain_size+2, 2))
    for depth in range(d):
        for lhs in gamma_B_counts[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_B_counts[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                j_model[depth, rhs_left_index, lhs_index,  1] += gamma_B_counts[depth][lhs][rhs]
                j_model[depth, rhs_left_index, :-1,  0] += gamma_A_counts[depth][lhs][rhs] * \
                                                         gamma_star_plus[depth, :, lhs_index]
    if normalize:
        return _normalize_a_tensor(j_model)
    return j_model

def _calc_a_model(gamma_counts, gamma_stars, d, abp_domain_size, normalize=False):
    gamma_star_plus, preterm_marginal_distr = gamma_stars
    gamma_A_counts, gamma_B_counts = gamma_counts
    a_model = np.zeros((d, abp_domain_size+2, abp_domain_size+2, abp_domain_size+2))
    for depth in range(d):
        for lhs in gamma_A_counts[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_A_counts[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                a_model[depth, :-1, rhs_left_index, lhs_index] += gamma_star_plus[depth, :, lhs_index] * \
                                                                 gamma_A_counts[depth][lhs][rhs]
    if normalize:
        return _normalize_a_tensor(a_model)
    return a_model

def _calc_b_models(gamma_counts, d, abp_domain_size, normalize=False):
    gamma_A_counts, gamma_B_counts = gamma_counts
    b_j0_model = np.zeros((d, abp_domain_size+2, abp_domain_size+2, abp_domain_size+2))
    b_j1_model = np.zeros((d, abp_domain_size+2, abp_domain_size+2, abp_domain_size+2))
    for depth in range(d):
        for lhs in gamma_A_counts[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_A_counts[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                rhs_right_index = int(rhs[1].symbol())
                # print(rhs_left_index, rhs_right_index)
                b_j0_model[depth, lhs_index, rhs_left_index, rhs_right_index] = gamma_A_counts[depth][lhs][rhs]
                b_j1_model[depth, lhs_index, rhs_left_index, rhs_right_index] = gamma_B_counts[depth][lhs][rhs]
    if normalize:
        return _normalize_a_tensor(b_j0_model), _normalize_a_tensor(b_j1_model)
    return b_j0_model, b_j1_model

def _calc_p_model(gamma_stars, d, abp_domain_size, normalize=False):
    gamma_star_plus, preterm_marginal_distr = gamma_stars
    # p_model = np.zeros((d, abp_domain_size+1, abp_domain_size+1))
    p_model = np.zeros((abp_domain_size + 2, abp_domain_size + 2))  # no depth
    for depth in range(d):
        for lhs in range(0,abp_domain_size+1):
            for rhs in range(0,abp_domain_size+1):
                # p_model[depth, lhs, rhs] = (gamma_star_plus[depth][lhs][rhs]) * preterm_marginal_distr[rhs]
                p_model[lhs, rhs] += (gamma_star_plus[depth][lhs][rhs]) * preterm_marginal_distr[rhs]
    if normalize:
        return _normalize_a_tensor(p_model)
    return p_model

def _calc_w_model(pcfg_counts, abp_domain_size, lex_size, normalize=False):
    w_model = np.zeros((abp_domain_size+2, lex_size))
    for lhs in pcfg_counts:
        lhs_index = int(lhs.symbol())
        for rhs in pcfg_counts[lhs]:
            if all([nltk.grammar.is_terminal(x) for x in rhs]) and len(rhs) == 1 and rhs[0] != '-ROOT-':
                w_model[lhs_index][int(rhs[0])] = pcfg_counts[lhs][rhs]
            elif all([nltk.grammar.is_terminal(x) for x in rhs]) and len(rhs) == 1 and rhs[0] == '-ROOT-':
                w_model[lhs_index][0] = pcfg_counts[lhs][rhs]
    if normalize:
        return _normalize_a_tensor(w_model)
    return w_model

def _normalize_a_tensor(tensor):
    return tensor / (np.sum(tensor, axis=-1, keepdims=True) + 1e-10)  # to supress zero division warning

def _inc_counts(model, ref_model, inc=1):
    if isinstance(model, list):
        for depth in range(len(model)):
            model[depth].pairCounts += ref_model[depth] * inc
    else:
        model.pairCounts += ref_model * inc

def pcfg_increment_counts(hid_seq, sent, models, inc=1, J=25, normalize=False, RB_init=False):
    d = len(models.A)
    d = d + 1  # calculate d+1 depth models for all pseudo count models, but not using them in _inc_counts
    abp_domain_size = models.A[0].dist.shape[0] - 2
    lex_size = models.lex.dist.shape[-1]
    if not RB_init:
        pcfg, pcfg_counts = translate_through_pcfg([(hid_seq, sent)],d, abp_domain_size)
    else:
        pcfg, pcfg_counts = translate_through_pcfg(RB_TREES*1000, d, abp_domain_size)
    nonterms = _build_nonterminals(abp_domain_size)
    delta_A, delta_B = _calc_delta(pcfg, J, abp_domain_size, d, nonterms)
    # print(delta_A, delta_B)
    gamma_A, gamma_B, gamma_A_counts, gamma_B_counts = _calc_gamma((delta_A, delta_B),pcfg, pcfg_counts,d)
    # print(_calc_expected_counts((gamma_A, gamma_B), pcfg, J, d, abp_domain_size))
    gamma_star, preterm_marginal_distr = _calc_expected_counts((gamma_A, gamma_B), pcfg_counts, J, d, abp_domain_size)
    # print("F")
    pseudo_F = _calc_f_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize)
    _inc_counts(models.F, pseudo_F, inc)
    # print(_calc_f_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize))
    # print("J")
    pseudo_J = _calc_j_model((gamma_A_counts, gamma_B_counts),(gamma_star, preterm_marginal_distr),d,abp_domain_size,normalize)
    _inc_counts(models.J, pseudo_J, inc)
    # print(_calc_j_model((gamma_A_counts, gamma_B_counts),(gamma_star, preterm_marginal_distr),d,abp_domain_size,normalize))
    # print("A")
    pseudo_A = _calc_a_model((gamma_A_counts, gamma_B_counts), (gamma_star, preterm_marginal_distr), d, abp_domain_size,normalize)
    _inc_counts(models.A, pseudo_A, inc)
    # print(_calc_a_model((gamma_A_counts, gamma_B_counts), (gamma_star, preterm_marginal_distr), d, abp_domain_size,normalize))
    # print("B")
    pseudo_B = _calc_b_models((gamma_A_counts, gamma_B_counts), d, abp_domain_size,normalize)
    _inc_counts(models.B_J0, pseudo_B[0], inc)
    _inc_counts(models.B_J1, pseudo_B[1], inc)
    # print(_calc_b_models((gamma_A_counts, gamma_B_counts), d, abp_domain_size,normalize))
    # print("P")
    pseudo_P = _calc_p_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize)
    _inc_counts(models.pos, pseudo_P, inc)
    # print(_calc_p_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize))
    # print("W")
    pseudo_W = _calc_w_model(pcfg_counts, abp_domain_size, lex_size, normalize)
    _inc_counts(models.lex, pseudo_W, inc)
    # print(_calc_w_model(pcfg_counts, abp_domain_size, lex_size, normalize))

def main():
    tree = ["-::ACT0/AWA0::+::POS1::1 -::ACT2/AWA2::+::POS1::1 +::ACT1/AWA2::-::POS2::2",
            "-::ACT0/AWA0::+::POS1::1 -::ACT2/AWA2::-::POS2::2"]
    # tree = ['-::ACT0/AWA0::+::POS1::1 -::ACT1/AWA2::+::POS1::1 +::ACT1/AWA2::-::POS2::2',
    # '-::ACT0/AWA0::+::POS1::1 -::ACT1/AWA2::-::POS2::2',
    # '-::ACT0/AWA0::+::POS1::1 -::ACT1/AWA1::-::POS1::1 -::ACT1/AWA2::-::POS2::2',
    # '-::ACT0/AWA0::+::POS1::1 -::ACT1/AWA2::-::POS2::2',
    # ]
    # tree_processed = full_chain_convert(tree, depth=2)
    abp_domain_size = 2
    d = 2
    J = 50
    normalize = False
    nonterms = _build_nonterminals(abp_domain_size)
    lex_size = 4
    # print(tree_processed)
    # print(tree_processed.productions())

    pcfg, pcfg_counts= translate_through_pcfg(tree*1000, d, abp_domain_size)
    print("PCFG")
    print(pcfg)
    print(pcfg_counts)
    print("DELTAs")
    # print(_calc_delta(pcfg, J, abp_domain_size, d,nonterms)[0], '\n',_calc_delta(pcfg, J, abp_domain_size, d,nonterms)[1])
    delta_A, delta_B = _calc_delta(pcfg, J, abp_domain_size, d, nonterms)
    print(delta_A, delta_B)
    gamma_A, gamma_B, gamma_A_counts, gamma_B_counts = _calc_gamma((delta_A, delta_B),pcfg, pcfg_counts,d)
    print("GAMMAs")
    print(gamma_A)
    print(gamma_B)
    print(gamma_A_counts)
    print(gamma_B_counts)
    print("GAMMA stars")
    gamma_star, preterm_marginal_distr = _calc_expected_counts((gamma_A, gamma_B), pcfg_counts, J, d, abp_domain_size)
    print(gamma_star,preterm_marginal_distr)
    print("F")
    print(_calc_f_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize))
    print("J")
    print(_calc_j_model((gamma_A_counts, gamma_B_counts),(gamma_star, preterm_marginal_distr),d,abp_domain_size,normalize))
    print("A")
    print(_calc_a_model((gamma_A_counts, gamma_B_counts), (gamma_star, preterm_marginal_distr), d, abp_domain_size,normalize))
    print("B")
    print(_calc_b_models((gamma_A_counts, gamma_B_counts), d, abp_domain_size,normalize))
    print("P")
    print(_calc_p_model((gamma_star, preterm_marginal_distr),d,abp_domain_size, normalize))
    print("W")
    print(_calc_w_model(pcfg_counts, abp_domain_size, lex_size, normalize))
if __name__ == '__main__':
    main()