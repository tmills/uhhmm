#!/usr/bin/env python3
from nltk.corpus import ptb
import re

"""
slicing penn treebank to extract sentences of less than N words and
output the gold standard trees.
"""

parsed_sents = ptb.parsed_sents()
with open('../test.txt', 'w') as fi:
    for index, sent in enumerate(ptb.tagged_sents()):
        count = 0
        for word, tag in sent:
            if 'NONE' not in tag:
                count += 1
        if count <= 20:
            tree = parsed_sents[index]
            for pos in tree.treepositions('leaves'):
                tree[pos] = tree[pos].lower()
            # tree.collapse_unary(collapsePOS=True)
            tree = str(tree).replace('\n', '')
            tree = re.sub('\s+', ' ', tree)
            print(tree, file=fi)