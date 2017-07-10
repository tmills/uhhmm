#!/usr/bin/env python3
import re
from nltk import tree
import string
"""
slicing penn treebank to extract sentences of less than N words and
output the gold standard trees.
the treebank file must a single file with all the empty nodes and traces removed from the trees
this outputs trees, and one can use make item to convert into tagwords or words

this is needed because the punctuations are used in inference but not in counting word number
"""
sent_length = 20  # the maximum number of words in a sentence

ptb_tree_file = '/home/jin.544/project_space/jin/corpora/ptb_trees/linetrees/wsj_full.linetrees'

ptb_puncs = [',', '.', ':', 'LRB', 'RRB', '$', '#', '``', "''"]

def bracket_check(word):
    return re.match('[LR][A-Z]B', word)

with open('genmodel/ptb_{}.linetrees'.format(sent_length), 'w') as fi, open(ptb_tree_file) as ptb:
    for index, line in enumerate(ptb):
        line = line.strip()
        this_tree = tree.Tree.fromstring(line)
        terminals = this_tree.pos()
        length = len(terminals)
        for word, pos in terminals:
            if pos in ptb_puncs:
                length -= 1
        if length <= sent_length and length > 0:
            print(line, file=fi)