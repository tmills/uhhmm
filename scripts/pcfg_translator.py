from left_corner2normal_tree_converter import full_chain_convert
import nltk
import numpy as np
from copy import deepcopy
"""
this file is for translating sequences of states to pcfg counts and back to uhhmm counts
the main function is translate_through_pcfg
"""

# class RecursiveProduction(nltk.grammar.Production):
#     def __init__(self, lhs, rhs=None, recur=0):
#         if isinstance(lhs, nltk.grammar.Production):
#             super().__init__(lhs.lhs(), lhs.rhs())
#         elif isinstance(lhs, nltk.grammar.Nonterminal):
#             super().__init__(lhs, rhs)
#         assert isinstance(recur, int)
#         self.recur = recur
#
#     def recur_value(self):
#         return self.recur
#
#     def recur_type(self):
#         if self.recur > 1:
#             return '+'
#         elif self.recur == 0:
#             return '0'
#         else:
#             return '*'
#
#     def __eq__(self, other):
#         return (type(self) == type(other) and self._symbol == other._symbol) or (type(other)==int and self._symbol == other)
#
#     def __str__(self):
#         """
#         Return a verbose string representation of the ``Production``.
#
#         :rtype: str
#         """
#         result = '%s -> ' % nltk.grammar.unicode_repr(self._lhs)
#         result += " ".join(nltk.grammar.unicode_repr(el) for el in self._rhs)
#         result += " "+'R'+str(self.recur)
#         return result

def translate_through_pcfg(seqs_of_states, depth, abp_domain_size):
    trees = []
    for seq in seqs_of_states:
        tree = full_chain_convert(seq, depth)
        trees.append(tree)
    count_dict = extract_counts(trees, abp_domain_size)
    return count_dict

def extract_counts(trees, abp_domain_size):
    # nonterms= _build_nonterminals(abp_domain_size)
    # count_dict_non_term_rules = {}
    # count_dict_term_rules = {}
    pcfg = {}
    for tree in trees:
        # rules = _extract_counts_single_tree(tree, nonterms)
        pcfg_rules = tree.productions()
        for rule in pcfg_rules:
            if rule.lhs() not in pcfg:
                pcfg[rule.lhs()] = {}
            pcfg[rule.lhs()][rule.rhs()] = pcfg[rule.lhs()].get(rule.rhs(), 0) + 1
        # for rule in rules:
        #     if rule.is_lexical():
        #         if rule.lhs() not in count_dict_term_rules:
        #             count_dict_term_rules[rule.lhs()] = {}
        #         if rule.rhs() not in count_dict_term_rules[rule.lhs()]:
        #             count_dict_term_rules[rule.lhs()][rule.rhs()] = 1
        #         else:
        #             count_dict_term_rules[rule.lhs()][rule.rhs()] += 1
        #     else:
        #         if rule.lhs() not in count_dict_non_term_rules:
        #             count_dict_non_term_rules[rule.lhs()] = {}
        #         if rule.rhs() not in count_dict_non_term_rules[rule.lhs()]:
        #             count_dict_non_term_rules[rule.lhs()][rule.rhs()] = {}
        #         if rule.recur_value() not in count_dict_non_term_rules[rule.lhs()][rule.rhs()]:
        #             count_dict_non_term_rules[rule.lhs()][rule.rhs()][rule.recur_value()] = 1
        #         else:
        #             count_dict_non_term_rules[rule.lhs()][rule.rhs()][rule.recur_value()] += 1
    pcfg_counts = deepcopy(pcfg)
    for lhs in pcfg:
        total = sum(pcfg[lhs].values())
        for rhs in pcfg[lhs]:
            pcfg[lhs][rhs] /= total
    return pcfg, pcfg_counts

# def _extract_counts_single_tree(tree, nonterminals):
#     final_productions = []
#     # get single layer productions:
#     ps = tree.productions()
#     for p in ps:
#         if p.is_lexical():
#             final_productions.append(RecursiveProduction(p, recur=0))
#         elif p.is_nonlexical():
#             final_productions.append(RecursiveProduction(p, recur=1))
#     # get recursive layer productions:
#     for subtree in tree.subtrees():
#         if subtree.height() <=3:
#             continue
#         elif subtree[0].height() > 2:
#             # print(subtree.pos(), subtree[0])
#             pos = subtree.pos()
#             rhs = (nonterminals[int(pos[0][1])], nonterminals[-1])
#             lhs = nonterminals[int(subtree.label())]
#             final_productions.append(RecursiveProduction(lhs, rhs, subtree[0].height()-1))
#     return final_productions


def _build_nonterminals(abp_domain_size):
    # we build the 0 nonterminal just for convenience. it is not used.
    return nltk.grammar.nonterminals(','.join([str(x) for x in range(0, abp_domain_size+1)])+',...' )

def _calc_delta(pcfg, i, abp_domain_size, d, nonterminals):
    delta = np.zeros((2, i, abp_domain_size+1, d))  # the delta model. s * d * i
    for a_index in range(abp_domain_size + 1):
        if a_index == 0:
            continue
        a = nonterminals[a_index]
        lexical_sum = sum([items[1] for items in pcfg[a].items() if len(items[0]) == 1 and
                           nltk.grammar.is_terminal(items[0][0])])
        delta[0, 1:, a_index, :] = lexical_sum
        delta[1, 1:, a_index, :] = lexical_sum
    for i_index in range(2, i):
        for a_index in range(abp_domain_size+1):  # the category labels correspond to their true labels. there is no 0
            if a_index == 0:
                continue
            a = nonterminals[a_index]
            for depth in range(d):
                nonterm_sum_a = 0
                nonterm_sum_b = 0
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
    return delta[0, -1,...].T, delta[1,-1,...].T

def _calc_gamma(deltas, pcfg_counts, d):
    delta_A, delta_B = deltas
    gamma_As, gamma_Bs = [], []
    for depth in range(d):
        gamma_As.append({})
        gamma_Bs.append({})
        for lhs in pcfg_counts:
            for rhs in pcfg_counts[lhs]:
                if any([nltk.grammar.is_terminal(x) for x in rhs]):
                    continue
                if lhs not in gamma_As[depth]:
                    gamma_As[depth][lhs] = {}
                    gamma_Bs[depth][lhs] = {}
                if rhs not in gamma_As[depth][lhs]:
                    gamma_As[depth][lhs][rhs] = 0
                    gamma_Bs[depth][lhs][rhs] = 0
                gamma_As[depth][lhs][rhs] = pcfg_counts[lhs][rhs] * delta_A[depth][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())] / delta_A[depth][int(lhs.symbol())]
                if depth + 1 < d:
                    gamma_Bs[depth][lhs][rhs] = pcfg_counts[lhs][rhs] * delta_A[depth+1][int(rhs[0].symbol())] * \
                                            delta_B[depth][int(rhs[1].symbol())] / delta_B[depth][int(lhs.symbol())]
    return gamma_As, gamma_Bs

def _calc_expected_counts(gammas, pcfg_counts, J, d, abp_domain_size):
    gamma_As, gamma_Bs = gammas
    gamma_star_plus = np.zeros((J, d, abp_domain_size+1, abp_domain_size+1))
    gamma_preterminal = np.zeros(abp_domain_size+1)
    for lhs in pcfg_counts:
        for rhs in pcfg_counts[lhs]:
            if all([nltk.grammar.is_terminal(x) for x in rhs]):
                lhs_index = int(lhs.symbol())
                gamma_preterminal[lhs_index] += 1
    for j in range(J):
        for depth in range(d):
            for lhs in gamma_As[depth]:
                lhs_index = int(lhs.symbol())
                for rhs in gamma_As[depth][lhs]:
                    a_prime = int(rhs[0].symbol())
                    if j == 0:
                        gamma_star_plus[j, depth, lhs_index, a_prime] += gamma_Bs[depth][lhs][rhs]
                    else:
                        gamma_star_plus[j, depth, 1:, a_prime] += gamma_star_plus[j-1, depth, 1:, lhs_index] \
                                                                 * gamma_As[depth][lhs][rhs]
    gamma_star_plus_final = np.sum(gamma_star_plus, axis=0)
    print(gamma_star_plus_final.shape)
    return gamma_star_plus_final, gamma_preterminal

def _calc_f_model(gamma_stars, d, abp_domain_size):
    gamma_star_plus, gamma_preterminal = gamma_stars
    f_model = np.zeros((d, abp_domain_size+1, abp_domain_size+1, 2))
    for depth in range(d):
        for lhs in range(1, abp_domain_size+1):
            for rhs in range(1, abp_domain_size+1):
                if lhs != rhs:
                    f_model[depth, lhs, rhs, 1] = gamma_star_plus[depth, lhs, rhs] * gamma_preterminal[lhs]
                else:
                    f_model[depth, lhs, rhs, 0] = gamma_preterminal[lhs]
                    f_model[depth, lhs, rhs, 1] = gamma_star_plus[depth, lhs, rhs] * gamma_preterminal[lhs]
    return f_model

def _calc_j_model(gammas, gamma_stars, d, abp_domain_size):
    gamma_star_plus, gamma_preterminal = gamma_stars
    gamma_As, gamma_Bs = gammas
    j_model = np.zeros((d, abp_domain_size+1, abp_domain_size+1, 2))
    for depth in range(d):
        for lhs in gamma_Bs[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_Bs[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                j_model[depth, lhs_index, rhs_left_index, 1] += gamma_Bs[depth][lhs][rhs]
                j_model[depth, 1:, rhs_left_index, 0] += gamma_As[depth][lhs][rhs] * (gamma_star_plus[depth, 1:, lhs_index]
                                                                                     + gamma_preterminal[lhs_index])
    return j_model

def _calc_a_model(gammas, gamma_stars, d, abp_domain_size):
    gamma_star_plus, gamma_preterminal = gamma_stars
    gamma_As, gamma_Bs = gammas
    a_model = np.zeros((d, abp_domain_size+1, abp_domain_size+1, abp_domain_size+1))
    for depth in range(d):
        for lhs in gamma_As[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_As[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                a_model[depth, 1:, rhs_left_index, lhs_index] += gamma_star_plus[depth, 1:, lhs_index] * \
                                                                gamma_As[depth][lhs][rhs]
    return a_model

def _calc_b_models(gammas, d, abp_domain_size):
    gamma_As, gamma_Bs = gammas
    b_j0_model = np.zeros((d, abp_domain_size+1, abp_domain_size+1, abp_domain_size+1))
    b_j1_model = np.zeros((d, abp_domain_size + 1, abp_domain_size + 1, abp_domain_size + 1))
    for depth in range(d):
        for lhs in gamma_As[depth]:
            lhs_index = int(lhs.symbol())
            for rhs in gamma_As[depth][lhs]:
                rhs_left_index = int(rhs[0].symbol())
                rhs_right_index = int(rhs[0].symbol())
                b_j0_model[depth, lhs_index, rhs_left_index, rhs_right_index] = gamma_As[depth][lhs][rhs]
                b_j1_model[depth, lhs_index, rhs_left_index, rhs_right_index] = gamma_Bs[depth][lhs][rhs]
    return b_j0_model, b_j1_model

def main():
    tree = "-::ACT0/AWA0::+::POS1::a -::ACT1/AWA2::+::POS1::a +::ACT1/AWA2::-::POS2::b"
    tree_processed = full_chain_convert(tree, depth=2)
    abp_domain_size = 2
    d = 2
    J = 50
    nonterms = _build_nonterminals(abp_domain_size)
    print(tree_processed)
    print(tree_processed.productions())
    # print(len(_extract_counts_single_tree(tree_processed, nonterms)))
    # assert len(_extract_counts_single_tree(tree_processed, nonterms)) == 21
    # for t in tree.subtrees():
    #     if t != tree:
    #         print(t)
    #         print(t.height())
    #         for x in t.treepositions():
    #             print(t[x],x)
    pcfg, pcfg_counts= translate_through_pcfg([tree], d, abp_domain_size)
    print("PCFG")
    print(pcfg)
    print("DELTAs")
    print(_calc_delta(pcfg, 100, abp_domain_size, d,nonterms)[0], '\n',_calc_delta(pcfg, J, abp_domain_size, d,nonterms)[1])
    delta_A, delta_B = _calc_delta(pcfg, J, abp_domain_size, d, nonterms)
    gamma_A, gamma_B = _calc_gamma((delta_A, delta_B),pcfg_counts, d)
    print("GAMMAs")
    print(gamma_A[0])
    print(gamma_A[1])
    print(gamma_B[0])
    print(gamma_B[1])
    print("GAMMA stars")
    print(_calc_expected_counts((gamma_A, gamma_B), pcfg_counts, J, d, abp_domain_size))
    gamma_star, gamma_term = _calc_expected_counts((gamma_A, gamma_B), pcfg_counts, J, d, abp_domain_size)
    print("F")
    print(_calc_f_model((gamma_star, gamma_term),d,abp_domain_size))
    print("J")
    print(_calc_j_model((gamma_A,gamma_B),(gamma_star, gamma_term),d,abp_domain_size))
    print("A")
    print(_calc_a_model((gamma_A,gamma_B), (gamma_star, gamma_term), d, abp_domain_size))
    print("B")
    print(_calc_b_models((gamma_A,gamma_B), d, abp_domain_size))
if __name__ == '__main__':
    main()