from left_corner2normal_tree_converter import full_chain_convert
import nltk
"""
this file is for translating sequences of states to pcfg counts and back to uhhmm counts
the main function is translate_through_pcfg
"""

class RecursiveProduction(nltk.grammar.Production):
    def __init__(self, lhs, rhs=None, recur=0):
        if isinstance(lhs, nltk.grammar.Production):
            super().__init__(lhs.lhs(), lhs.rhs())
        elif isinstance(lhs, nltk.grammar.Nonterminal):
            super().__init__(lhs, rhs)
        assert isinstance(recur, int)
        self.recur = recur

    def recur_value(self):
        return self.recur

    def recur_type(self):
        if self.recur > 1:
            return '+'
        elif self.recur == 0:
            return '0'
        else:
            return '*'

    def __str__(self):
        """
        Return a verbose string representation of the ``Production``.

        :rtype: str
        """
        result = '%s -> ' % nltk.grammar.unicode_repr(self._lhs)
        result += " ".join(nltk.grammar.unicode_repr(el) for el in self._rhs)
        result += " "+'R'+str(self.recur)
        return result

def translate_through_pcfg(seqs_of_states, depth, abp_domain_size):
    trees = []
    for seq in seqs_of_states:
        tree = full_chain_convert(seq, depth)
        trees.append(tree)
    count_dict = extract_counts(trees, abp_domain_size)
    return count_dict

def extract_counts(trees, abp_domain_size):
    nonterms= _build_nonterminals(abp_domain_size)
    count_dict_non_term_rules = {}
    count_dict_term_rules = {}
    pcfg = {}
    for tree in trees:
        rules = _extract_counts_single_tree(tree, nonterms)
        pcfg_rules = tree.productions()
        for rule in pcfg_rules:
            if rule.lhs() not in pcfg:
                pcfg[rule.lhs()] = {}
            pcfg[rule.lhs()][rule.rhs()] = pcfg[rule.lhs()].get(rule.rhs(), 0) + 1
        for rule in rules:
            if rule.is_lexical():
                if rule.lhs() not in count_dict_term_rules:
                    count_dict_term_rules[rule.lhs()] = {}
                if rule.rhs() not in count_dict_term_rules[rule.lhs()]:
                    count_dict_term_rules[rule.lhs()][rule.rhs()] = 1
                else:
                    count_dict_term_rules[rule.lhs()][rule.rhs()] += 1
            else:
                if rule.lhs() not in count_dict_non_term_rules:
                    count_dict_non_term_rules[rule.lhs()] = {}
                if rule.rhs() not in count_dict_non_term_rules[rule.lhs()]:
                    count_dict_non_term_rules[rule.lhs()][rule.rhs()] = {}
                if rule.recur_value() not in count_dict_non_term_rules[rule.lhs()][rule.rhs()]:
                    count_dict_non_term_rules[rule.lhs()][rule.rhs()][rule.recur_value()] = 1
                else:
                    count_dict_non_term_rules[rule.lhs()][rule.rhs()][rule.recur_value()] += 1
    for lhs in pcfg:
        total = sum(pcfg[lhs].values())
        for rhs in pcfg[lhs]:
            pcfg[lhs][rhs] /= total
    return count_dict_term_rules, count_dict_non_term_rules, pcfg

def _extract_counts_single_tree(tree, nonterminals):
    final_productions = []
    # get single layer productions:
    ps = tree.productions()
    for p in ps:
        if p.is_lexical():
            final_productions.append(RecursiveProduction(p, recur=0))
        elif p.is_nonlexical():
            final_productions.append(RecursiveProduction(p, recur=1))
    # get recursive layer productions:
    for subtree in tree.subtrees():
        if subtree.height() <=3:
            continue
        elif subtree[0].height() > 2:
            # print(subtree.pos(), subtree[0])
            pos = subtree.pos()
            rhs = (nonterminals[int(pos[0][1])-1], nonterminals[-1])
            lhs = nonterminals[int(subtree.label())-1]
            final_productions.append(RecursiveProduction(lhs, rhs, subtree[0].height()-1))
    return final_productions


def _build_nonterminals(abp_domain_size):
    return nltk.grammar.nonterminals(','.join([str(x) for x in range(1, abp_domain_size+1)])+',...' )

def main():
    tree = "-::ACT0/AWA0::+::POS2::mom -::ACT1/AWA5::-::POS5::bit -::ACT5/AWA1::-::POS1::the -::ACT4/AWA1::+::POS2::hold " \
           "+::ACT4/AWA1::-::POS1::in -::ACT4/AWA1::-::POS1::it -::ACT3/AWA3::+::POS5::, +::ACT3/AWA3::+::POS2::eve +::ACT3/AWA3::-::POS3::."
    tree_processed = full_chain_convert(tree, depth=1)
    nonterms = _build_nonterminals(5)
    print(tree_processed)
    print(tree_processed.productions())
    print(len(_extract_counts_single_tree(tree_processed, nonterms)))
    assert len(_extract_counts_single_tree(tree_processed, nonterms)) == 21
    # for t in tree.subtrees():
    #     if t != tree:
    #         print(t)
    #         print(t.height())
    #         for x in t.treepositions():
    #             print(t[x],x)
    counts = translate_through_pcfg([tree], 1, 5)
    print(counts[0])
    print(counts[1])
    print(counts[2])


if __name__ == '__main__':
    main()