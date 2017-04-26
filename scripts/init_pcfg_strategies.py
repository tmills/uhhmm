from nltk import tree
import random

# strategies for initializing the ints corpus
def left_branching_strategy(line, abp_domain_size):
    new_tree = None
    for word in line:
        new_lex = tree.Tree(str(random.randint(1, abp_domain_size)), [word])
        if new_tree:
            new_tree = tree.Tree(str(random.randint(1, abp_domain_size)), [new_tree, new_lex])
        else:
            new_tree = new_lex
    return new_tree

def right_branching_strategy(line, abp_domain_size):
    new_tree = None
    for word in line[::-1]:
        new_lex = tree.Tree(str(random.randint(1, abp_domain_size)), [word])
        if new_tree:
            new_tree = tree.Tree(str(random.randint(1, abp_domain_size)), [new_lex, new_tree])
        else:
            new_tree = new_lex
    return new_tree

STRATEGY_STRINGS = {'left_branching':left_branching_strategy, 'right_branching':right_branching_strategy}
