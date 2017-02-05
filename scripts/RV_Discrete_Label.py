"""
Wrapper module for scipy.stats.rv_discrete that allows:

- Indexing by any hashable python object
- Concatenation of discrete distributions
- Maintenance of updatable weights/counts for each item, on which the distribution is based

@author: Cory Shain
"""

import copy, numpy as np
from numpy import inf
from scipy.stats import rv_discrete

class rv_discrete_label(object):
  def __init__(this, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, extradoc=None, seed=None, stick=1.0):
    this.a = a
    this.b = b
    this.name = name
    this.badvalue = badvalue
    this.moment_tol = moment_tol
    this.labels = values[0]
    this.weights = values[1]
    this.inc = inc
    this.longname = longname
    this.shapes = shapes
    this.extradoc = extradoc
    this.seed = seed
    this.stick = 1.0
    this.dist = None
    this.get_labels = np.vectorize(lambda this, x: this.labels[x])
    this.refresh_dist()

  def stringify_label(this, lab):
    lab = str(lab)
    if lab.startswith('('): lab = lab[1:]
    if lab.endswith(')'): lab = lab[:-1]
    lab = lab.replace(',', '')
    lab = lab.replace("'", '')
    return lab

  def __str__(this):
    out = ''
    label_probs = []
    norm = this.weights.sum()
    for i in xrange(len(this.labels)):
      label_probs.append((this.labels[i], this.weights[i]))
    label_probs.sort(key=lambda x: x[1], reverse=True)
    for pair in label_probs:
      out += 'R %s : %s = %s\n' %(this.name, this.stringify_label(pair[0]), pair[1]/norm)
    return out
  
  def append(this, x, mixing=None):
    norm = this.weights.sum() + x.weights.sum()
    check = None
    if not mixing:
      mixing = float(x.weights.sum()) / (norm)
#      check = copy.copy(this.weights)
    assert mixing >= 0 and mixing <= 1, 'Invalid mixing parameter %s. 0 <= mixing <= 1.' %mixing 
    this.labels += x.labels
    this.weights *= ((1.0/this.weights.sum())*(1-mixing)*norm)
    this.weights = np.append(this.weights, x.weights * (1.0/x.weights.sum())*(mixing)*norm)
#    if check != None:
#      assert np.allclose(this.weights[:len(check)], check) and np.allclose(this.weights[len(check):], x.weights), 'Null mixing has not worked properly, input and output weights unequal.\n Input 1 = %s.\n Input 2 = %s.\n Output 1 = %s.\n Output 2 = %s.\n Mixing = %s.\n Sum weights input 1 = %s.\n Sum weights input 2 = %s.\n Sum weights input 1 + input 2 = %s.\n \n Sum weights output = %s.' %(check, x.weights, this.weights[:len(check)], this.weights[len(check):], mixing, check.sum(), x.weights.sum(), check.sum() + x.weights.sum(), this.weights.sum()) 
#      assert len(this.weights) == len(check) + len(x.weights), 'Problem with append operation. Output length differs from sum of inputs. Len input 1 = %s. Len input 2 = %s. Len output = %s.' %(len(check), len(x.weights), len(this.weights))
    this.refresh_dist()

  def rvs(this, size=1):
    assert len(this.labels) == len(this.weights), 'Prob model broken:\n Labels: %s\n Weights: %s' %(this.labels, this.weights)
    samples = this.dist.rvs(size=size)
    return this.get_labels(this, samples)

  def refresh_dist(this):
    assert len(this.labels) == len(this.weights), 'Lists of labels and probabilities have unequal lengths. Labels = %s. Probs = %s.' %(len(this.labels), len(this.weights))
    this.dist = rv_discrete(this.a,
                            this.b,
                            this.name,
                            this.badvalue,
                            this.moment_tol,
                            (range(len(this.labels)), this.weights / this.weights.sum()),
                            this.longname,
                            this.shapes,
                            this.extradoc,
                            this.seed)


