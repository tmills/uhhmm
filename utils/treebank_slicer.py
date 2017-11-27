#!/usr/bin/env python3
import re
from nltk import tree
import string
import argparse
"""
slicing penn treebank to extract sentences of less than N words and
output the gold standard trees.
the treebank file must a single file with all the empty nodes and traces removed from the trees
this outputs trees, and one can use make item to convert into tagwords or words

this is needed because the punctuations are used in inference but not in counting word number
"""
parser = argparse.ArgumentParser()
parser.add_argument('-n', required=True, type=int, help='the number of words at most in a sent')
parser.add_argument('-c', default='', type=str, help='the path to the full treebank file')
parser.add_argument('-ref', default='', type=str, help='the path to the referred file for '
                                                       'filtering')
args = parser.parse_args()
sent_length = args.n  # the maximum number of words in a sentence

if not args.c:
    ptb_tree_file = '/home/jin.544/project_space/jin/corpora/ptb_trees/linetrees/wsj_full.linetrees'
else:
    ptb_tree_file = args.c

ptb_puncs = [',', '.', ':', 'LRB', 'RRB', '$', '#', '``', "''"]

def bracket_check(word):
    return re.match('[LR][A-Z]B', word)

if args.ref:
    with open(args.ref) as ref:
        referred_sents = ref.readlines()
    output_fn = 'genmodel/ptb_{}.ref.linetrees'.format(sent_length)
else:
    referred_sents = []
    output_fn = 'genmodel/ptb_{}.linetrees'.format(sent_length)


with open(output_fn, 'w') as fi, open(ptb_tree_file) as ptb:
    for index, line in enumerate(ptb):
        this_line = line.strip()
        this_tree = tree.Tree.fromstring(this_line)
        terminals = this_tree.pos()
        length = len(terminals)
        for word, pos in terminals:
            if pos in ptb_puncs:
                length -= 1
        if length <= sent_length and length > 0:
            if not referred_sents or line not in referred_sents:
                print(this_line, file=fi)