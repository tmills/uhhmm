"""
Module for generating random PCFGs and sampling trees from them.

@author: Cory Shain
"""

import sys, math, copy, numpy as np
from numpy import inf
from RV_Discrete_Label import rv_discrete_label
import tree

"""
Class for generating and sampling from a random PCFG.

Uses weights sampled from a uniform distribution for calculating all distributions
except P(word|pre-terminal), which are generating using the stick-breaking construction
in order to model the long-tailed distribution of words in natural languages.

@param a_max: Number of active (A) category labels
@param b_max: Number of awaited (B) category labels
@param w_max: Number of words (W)
@param w_len_min: Minimum word length
@param w_len_max: Maximum word length
@param branching: Branching preference (i.e. left-branching probability; 0 <= branching <= 1)
@param recursion: Recursion probability (0 <= recursion < 1)
@param recursion_bound: Maximum depth of recursion allowed during tree sampling.
@param termination: Branch termination (bottom-out) probability (0 < termination <= 1)
@param one_wrd: Probability of generating a one-word sentence (0 <= one_wrd <= 1)
@param alpha_coef: Generate probs with stick-breaking construction parameterized by |outcomes| * alpha_coef. Lower values create more concentrated distributions. If <None>, probabilities will be created from randomly-sampled weights rather than Dirichlet processes. Also accepts the string value 'rand', which generates random concentration coefficients for each distribution in the model.
"""

class PCFG_Generator():
  def __init__(this, a_max=5, b_max=5, p_max=5, w_max=2000, w_len_min=1, w_len_max=15, branching=None, recursion=None, recursion_bound=None, termination=None, one_wrd=None, alpha_coef=None):
    this.a_max = a_max
    this.b_max = b_max
    this.p_max = p_max
    this.w_max = w_max
    this.w_len_min = w_len_min
    this.w_len_max = w_len_max
    this.w_len_mean = (w_len_min+w_len_max) / 2
    this.branching = branching
    if recursion == 1:
      # Probability 1 disallowed, padding recursion param
      this.recursion = 0.9999999999
    else: this.recursion = recursion
    this.recursion_bound = recursion_bound
    if termination == 0.0:
      # Zero termination probability disallowed, padding termination param
      this.termination = 0.00000000001
    else: this.termination = termination
    this.one_wrd = one_wrd
    this.alpha_coef = alpha_coef
    this.lex = set([''])
    this.model = {}

    this.generate_lex()
    this.generate_model()

  def __str__(this):
    out = ''
    for c in this.model:
      if not type(this.model[c]) == dict:
    #    out += str(this.model[c].labels) + '\n'
    #    out += str(this.model[c].weights) + '\n'
        out += str(this.model[c])
    return out

  """
  Generates a lexicon.
  """
  def generate_lex(this):
    alph = 'abcdefghijklmnopqrstuvwxyz'
    for i in xrange(this.w_max):
      w = ''
      while w in this.lex:
        w_len = np.random.normal(this.w_len_mean, float(this.w_len_max-this.w_len_mean)/2)
        if w_len < this.w_len_min: w_len = this.w_len_min
        elif w_len > this.w_len_max: w_len = this.w_len_max
        while len(w) < w_len:
          w += alph[np.random.randint(0, len(alph))]
          add_char = np.random.uniform() < float(this.w_len_max - 1) / this.w_len_max
      this.lex.add(w)
    this.lex.remove('')

  """
  Adds a condition to the probability model (or updates a condition if it already exists).

  @param c: Condition name
  @param outcome_list: List of outcomes dependent on the condition for which to generate probabilities
  @param model: The model (dictionary) to which to add the condition (defaults to using the instance model)
  @param mixing: Real number s.t. 0 <= mixing <= 1. Represents the portion of probability mass to allocate to the new outcomes in the list (the remainder will be allocated to the existing outcomes). If left blank, weights will simply be appended for the new outcomes with no reallocation.
  @param alpha: Hyperparameter for Dirichlet process (DP) if used for generating the distribution. If left blank, DP will not be used.

  @return: void
  """
  def add_condition(this, c, outcome_list, model=None, mixing=None):
    if not model:
      model = this.model
    if mixing != None:
      assert c in model, 'Mixing parameter supplied but no existing model for %s to append to.' %c
    # Gives all prob mass to old outcomes.
    # Save space by skipping zero-probability new outcomes
    if mixing == 0.0:
      return
    dist = this.generate_rv_dist(c, outcome_list, this.alpha_coef)
    # Gives all prob mass to new outcomes
    # Save space by skipping zero-probability old outcomes
    if mixing == 1.0 or c not in model:
      model[c] = dist
      return
    model[c].append(dist, mixing=mixing)
   
  """
  Generates a random probability distribution for a discrete random variable.

  @param c: Condition name
  @param outcome_list: List of outcomes dependent on the condition for which to generate probabilities
  @param alpha: Hyperparameter for Dirichlet process (DP) if used for generating the distribution. If left blank, DP will not be used.
  @param stick_init: Portion of beta stick to sample from if using DP.

  @return: rv_discrete_label instance representing new distribution
  """
  def generate_rv_dist(this, c, outcome_list, alpha_coef=None):
    outcomes = outcome_list[:]
    outcome_count = len(outcomes)
    np.random.shuffle(outcomes)
    outcome_weights = np.zeros(outcome_count)
    
    if alpha_coef != None:
      if alpha_coef == 'rand':
        alpha_coef = np.random.rand()
      stick = 1.0
      for i in range(0, outcome_count):
        outcome_weights[i] = beta = np.random.beta(1,outcome_count*alpha_coef)*stick
        stick -= beta
    else:
      outcome_weights = np.random.rand(outcome_count)

    return rv_discrete_label(values=(outcomes, outcome_weights), name=c)

  """
  Generates PCFG probability model.
  """
  def generate_model(this):
    # Rule classes:
    #
    #   A -> A B : branching-neutral
    #   A -> A P : left-branching
    #   A -> P B : right-branching
    #   A -> P P : terminating
    #
    #   B -> A B : center-embedding; branching-neutral
    #   B -> A P : center-embedding; left-branching
    #   B -> P B : right-branching
    #   B -> P P : terminating
    #

    # Parameterization priority:
    #
    #   termination rate > recursion rate > branching rate
    #
    # Priority enforced by order of model construction
    # (lower priority first)

    a_list = ['ACT%s'%a for a in range(1,this.a_max+1)]
    b_list = ['AWA%s'%b for b in range(1,this.b_max+1)]
    p_list = ['POS%s'%p for p in range(1,this.p_max+1)]
    lex_list = list(this.lex)

    # Initialize root probabilities
    this.add_condition('T', a_list)
    this.add_condition('T', p_list, mixing=this.one_wrd)

    right_branching_rules = [(P,B) for P in p_list for B in b_list]
    left_branching_rules = [(A,P) for A in a_list for P in p_list]
    recursive_rules = [(A,B) for A in a_list for B in b_list]
    termination_rules = [(P,P) for P in p_list for P in p_list]

    this.model['depth_exceeded'] = {}
    this.model['terminate'] = {}

    for a in a_list:
      assert a not in this.model, 'Error: %s already in this.model.' %b
      assert a not in this.model['depth_exceeded'], 'Error: %s already in this.model["depth_exceeded"].' %a
      # Add right-branching rules
      this.add_condition(a, right_branching_rules)
      # Append left-branching rules, partitioning probability per branching param
      this.add_condition(a, left_branching_rules, mixing=this.branching)
      # Append recursive rules, partitioning probability per recursion param
      this.add_condition(a, recursive_rules, mixing=this.recursion)
      # Append termination rules, partitioning probability per termination param.
      terminate = this.generate_rv_dist(a, termination_rules, this.alpha_coef)
      this.model['terminate'][a] = copy.deepcopy(terminate)
      this.model[a].append(terminate, mixing=this.termination)
      # Copy entire A model to the "depth_exceeded" distribution, since all A-headed rules do not increase depth
      this.model['depth_exceeded'][a] = copy.deepcopy(this.model[a])

    for b in b_list:
      assert b not in this.model, 'Error: %s already in this.model.' %b
      assert b not in this.model['depth_exceeded'], 'Error: %s already in this.model["depth_exceeded"].' %b
      # Add right-branching rules
      # Copy right-branching probs to the "depths_exceeded" distribution for use when center-embedding disallowed
      rb = this.generate_rv_dist(b, right_branching_rules, this.alpha_coef)
      this.model['depth_exceeded'][b] = rb
      this.model[b] = copy.deepcopy(rb)
      # Append left-branching rules, partitioning probability per branching param.
      this.add_condition(b, left_branching_rules, mixing=this.branching)
      # Append recursive rules, partitioning probability per recursion param
      this.add_condition(b, recursive_rules, mixing=this.recursion)
      # Append termination rules, partitioning probability per termination param.
      terminate = this.generate_rv_dist(a, termination_rules, this.alpha_coef)
      this.model['terminate'][b] = copy.deepcopy(terminate)
      this.model[b].append(terminate, mixing=this.termination)
      # Copy non-center-embedding rules to the "depth_exceeded" distribution
      this.model['depth_exceeded'][b].append(terminate, mixing=this.termination)

    for p in p_list:
      # Use stick-breaking to model long-tailed distribution of words in natural languages
      this.add_condition(p, lex_list)

    #print('Model generated.')
  
  """
  Sample a batch of trees from the PCFG.

  @param n: Number of trees to sample.
  @param D: Maximum depth allowed.

  @return: List of tree objects.
  """
  def generate_trees(this, n, D=inf):
    trees = []
    for i in xrange(n):
      t = tree.Tree()
      this.generate_tree(t, D=D)
      trees.append(t)
      if i > 0 and i%1000 == 0:
        sys.stderr.write('%d trees sampled.\n'%i)
    return trees

  """
  Recursively sample a tree from the PCFG.

  @param t: Root tree object.
  @param d: Current embedding depth.
  @param r: Current recursion depth.
  @param rchild: Whether or not t is a right child

  @return: Tree object
  """
  def generate_tree(this, t, d=0, r=0, D=inf, rchild=False):
    r += 1
    if not t.c:
      t.c = this.model['T'].rvs()[0]
      this.generate_tree(t, d+1, r, D, rchild)
    else:
      if t.c[:4] in this.model:
        if not t.c.startswith('POS'):
          if r >= this.recursion_bound:
            ch = this.model['terminate'][t.c].rvs()
          elif d >= D:
            ch = this.model['depth_exceeded'][t.c].rvs()
          else:
            ch = this.model[t.c].rvs()
          lc = tree.Tree(ch[0][0])
          rc = tree.Tree(ch[1][0])
          if rchild:
            dl = d+1
          else: dl = d
          
          this.generate_tree(lc, dl, r, D, False)
          this.generate_tree(rc, d, r, D, True)
          t.ch = [lc, rc]
        else:
          ch = this.model[t.c].rvs()
          lc = tree.Tree(ch[0])
          this.generate_tree(lc, d, r, D, False)
          t.ch = [lc]

