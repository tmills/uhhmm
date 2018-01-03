import logging
import operator
import random

import nltk
from typing import List

THRESHOLD = 5e-2

class PCFGMedic:
    def __init__(self):
        pass

    @classmethod
    def cpr(cls, trees: List[nltk.tree.Tree], abp_domain_size):
        need_cpr, cpr_source, cpr_target = cls.check_counts(trees, abp_domain_size)
        if need_cpr:
            cls.apply_split(trees, cpr_target, cpr_source)

    @classmethod
    def check_counts(cls, trees : List[nltk.tree.Tree], abp_domain_size):
        category_counts = {}
        total_counts = 0
        for i in range(abp_domain_size + 1):
            if i != 0:
                category_counts[nltk.grammar.Nonterminal(str(i))] = 0
        for tree in trees:
            productions = tree.productions()
            for production in productions:
                if production.lhs().symbol() != '0':
                    assert production.lhs() in category_counts
                    category_counts[production.lhs()] += 1
                    total_counts += 1
        average_count = total_counts / (abp_domain_size - 1)
        need_cpr = False
        cpr_target = None
        cpr_source = None
        category_list = list(category_counts.items())
        category_list.sort(key=operator.itemgetter(1))
        logging.info('PCFG medic: lowest count category has {} and threshold is {}'.format(
            category_list[0][1], average_count * THRESHOLD))
        if category_list[0][1] < average_count * THRESHOLD:
            cpr_target = category_list[0][0]
            cpr_source = category_list[-1][0]
            count = category_list[0][1]
            logging.warning('Category {} has way too low count {}! CPR activated by '
                            'splitting category {}.'.format(
                    cpr_target, count, cpr_source))
            need_cpr = True
        return need_cpr, cpr_source, cpr_target

    @classmethod
    def apply_split(cls, trees: List[nltk.tree.Tree], cpr_target, cpr_source):
        cpr_target_label = cpr_target.symbol()
        cpr_source_label = cpr_source.symbol()
        for tree in trees:
            for position in tree.treepositions():
                if not isinstance(tree[position], str) and tree[position].label() == \
                        cpr_source_label:
                    if random.random() > 0.5:
                        tree[position].set_label(cpr_target_label)

# t = [nltk.tree.Tree.fromstring("(0 (2 (1 the) (2 dog)) (3 (1 chased) (2 (2 the) (3 cat))))")]
# PCFGMedic.cpr(t, 5)
# print(t)
