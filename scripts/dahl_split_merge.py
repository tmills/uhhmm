#!/usr/bin/env python3.4
import numpy as np
import logging
import copy
from scipy.special import gammaln

# sentence and word indices at given position in sequence
def indices (cum_length, ind):
    sent_ind = np.where(cum_length > ind)[0][0]
    word_ind = ind if sent_ind == 0 else ind - cum_length[sent_ind-1]
    return sent_ind, int(word_ind)

# log emission posterior normalizing constant
# assumes Multinomial-Dirichlet model
# used for lex | pos and pos | awa
def norm_const(counts, alpha):
    return sum(gammaln(counts+alpha)) - gammaln(sum(counts+alpha))

# helper function for unit tests
def print_hid_seqs(hid_seqs): 
    sent=0
    for b in hid_seqs:
        logging.debug("hid_seqs for sentence %d:" % sent)
        for a in b:
            logging.debug(a.g)
        sent = sent+1

# Sequentially Allocated Merge Split algorithm (Dahl 2005, section 4.1)
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.131.3712&rep=rep1&type=pdf 
def perform_split_merge_operation(models, sample, ev_seqs, params, iter, equal=False, ind = None):
    num_tokens = sum(map(len, ev_seqs))
    cum_length = np.cumsum(list(map(len, ev_seqs)))
    ## need to copy the models and sample, otherwise we'll just have another pointer
    new_models = copy.deepcopy(models)
    new_sample = copy.deepcopy(sample)

    if ind: # if using provided anchor indices during backtracking check (false by default)
      ind0 = ind[0]
      ind1 = ind[1]
    else:
      ind0 = np.random.randint(num_tokens) # anchor 0
      sent0_ind, word0_ind = indices(cum_length, ind0)
      pos0 = sample.hid_seqs[sent0_ind][word0_ind].g
      if equal: # propose merges and splits equally often (false by default)
          split = (np.random.binomial(1, .5) > .5)
          subset = np.array([]) 
          for ind in range(num_tokens):
              sent_ind, word_ind = indices(cum_length, ind)
              state = sample.hid_seqs[sent_ind][word_ind]
              if (((state.g == pos0) and split) or ((state.g != pos0) and not split)):
                  subset = np.append(subset, ind)
          ind1 = np.random.choice(subset) # anchor 1
      else: # propose a merge if the anchors have different tags, otherwise a split (favors merges)
          ind1 = np.random.randint(num_tokens) # anchor 1

    sent0_ind, word0_ind = indices(cum_length, ind0)
    pos0 = sample.hid_seqs[sent0_ind][word0_ind].g
    sent1_ind, word1_ind = indices(cum_length, ind1)
    pos1 = sample.hid_seqs[sent1_ind][word1_ind].g
    split = (pos0 == pos1) # true if splitting, false if merging

    alpha_lex = np.array(params['h'][0,1:])
    logging.debug("Lexical Dirichlet pseudocounts:")
    logging.debug(alpha_lex)
    alpha_pos_vec = models.pos.alpha * models.pos.beta[1:]
    if split: # Dirichlet pseudocounts for the original POS tag are split in half for the two new tags
        alpha_pos = np.array([0.5*alpha_pos_vec[pos0-1], 0.5*alpha_pos_vec[pos0-1]])
    else:
        alpha_pos = np.array([alpha_pos_vec[pos0-1], alpha_pos_vec[pos1-1]])
    logging.debug("POS Dirichlet pseudocounts:")
    logging.debug(alpha_pos)
        
    subset = np.array([], dtype=int) # indices with state pos0 or pos1
    for ind in range(num_tokens):
        sent_ind, word_ind = indices(cum_length, ind)
        state = sample.hid_seqs[sent_ind][word_ind]
        if ((state.g == pos0) or (state.g == pos1)):
            subset = np.append(subset, ind)     
    np.random.shuffle(subset)

    # indices with state pos0 or pos1, excluding anchors
    subset1 = np.delete(subset, np.where(subset == ind0)[0])
    subset1 = np.delete(subset1, np.where(subset1 == ind1)[0])
    
    sets = [[ind0], [ind1]] # sets of indices that go into the two POS tags
    if not split:
        st = {pos0: 0, pos1: 1} # binary indicator for the two POS tags that are being merged

    num_unique_lex = len(set(sum(ev_seqs, [])))
    lex_counts = np.zeros((2, num_unique_lex), dtype=int) # lexical counts for the two POS tags
    lex_counts[0, ev_seqs[sent0_ind][word0_ind]-1] = 1
    lex_counts[1, ev_seqs[sent1_ind][word1_ind]-1] = 1
    num_awa = models.pos.dist.shape[0] # number of awaited categories
    pos_counts = np.zeros((num_awa, 2), dtype=int) # POS counts for all awaited variable values
    awa0 = sample.hid_seqs[sent0_ind][word0_ind].b[0] # awaited value for anchor 0
    awa1 = sample.hid_seqs[sent1_ind][word1_ind].b[0] # awaited value for anchor 1
    pos_counts[awa0, 0] += 1
    pos_counts[awa1, 1] += 1
    norm_const_lex = np.array([norm_const(lex_counts[0], alpha_lex), norm_const(lex_counts[1], alpha_lex)])

    logprob_split = 0 # log probability of the new state assignments during the split
    for ind in subset1: # assign the non-anchor indices to the two POS tags one by one
        sent_ind, word_ind = indices(cum_length, ind)
        awa = sample.hid_seqs[sent_ind][word_ind].b[0]
        # new lexical and POS counts for the two possible state assignments:
        newcounts_lex = np.copy(lex_counts)
        newcounts_lex[:, ev_seqs[sent_ind][word_ind]-1] += 1
        newcounts_pos0 = np.copy(pos_counts[awa])
        newcounts_pos1 = np.copy(pos_counts[awa])
        newcounts_pos0[0] += 1
        newcounts_pos1[1] += 1
        logterms1 = np.array([norm_const(newcounts_lex[0], alpha_lex), norm_const(newcounts_lex[1], alpha_lex)]) - norm_const_lex
        logterms = logterms1 + np.array([norm_const(newcounts_pos0, alpha_pos), norm_const(newcounts_pos1, alpha_pos)])
        terms = [np.exp(logterms[0] - max(logterms)) * len(sets[0]), np.exp(logterms[1] - max(logterms)) * len(sets[1])]
        # new state assignment is random for a split and pre-determined for a merge:
        newstate = np.random.binomial(1, terms[1]/sum(terms)) if split else st[sample.hid_seqs[sent_ind][word_ind].g]
        logprob_split += logterms[newstate] - max(logterms) + np.log(len(sets[newstate])) - np.log(sum(terms))
        sets[newstate].append(ind)
        norm_const_lex[newstate] += logterms1[newstate]
        lex_counts[newstate] = newcounts_lex[newstate]
        pos_counts[awa] = newcounts_pos1 if (newstate == 1) else newcounts_pos0
    
    logging.debug("During split-merge the shape of root is %s and exp is %s" % (str(models.root[0].dist.shape), str(models.exp[0].dist.shape) ) )
    if split:
        logging.info("Performing split operation of pos tag %d at iteration %d" % (pos0,iter))
        from uhhmm import break_g_stick
        break_g_stick(new_models, new_sample, params) # add new pos tag variable
        pos1 = models.pos.dist.shape[1]-1
        # for the POS tag being split, assign half of the beta parameter to each of the two resulting tags
        new_models.pos.beta[pos0] = 0.5*new_models.pos.beta[pos0]
        new_models.pos.beta[pos1] = new_models.pos.beta[pos0]
        for ind in sets[1]: # reassign indices in set1 from pos0 to pos1
            sent_ind, word_ind = indices(cum_length, ind)
            state = new_sample.hid_seqs[sent_ind][word_ind]
            token = ev_seqs[sent_ind][word_ind]
            state.g = pos1
            new_models.pos.count(state.b[0], pos0, -1)
            new_models.pos.count(state.b[0], state.g, 1)
            new_models.lex.count(pos0, token, -1)
            new_models.lex.count(state.g, token, 1)
        logging.debug("During split the new shape of root is %s and exp is %s" % (str(new_models.root[0].dist.shape), str(new_models.exp[0].dist.shape) ) )
        assert new_models.pos.dist.shape[1] == new_models.root[0].dist.shape[1], "new pos dist size inconsistent with root: %d" % new_models.pos.dist.shape[1]  
    else:
        (pos0, pos1) = (min(pos0, pos1), max(pos0, pos1))
        if models.pos.dist.shape[1] <= 4:
            logging.warn("Tried to perform a merge with only %d state(s) left!" % (models.pos.dist.shape[1] - 2) )
            return models, sample
        logging.info("Performing merge operation between pos tags " + \
          "%d and %d at iteration %d" % (pos0, pos1, iter))
        for ind in range(num_tokens):
            sent_ind, word_ind = indices(cum_length, ind)
            state = new_sample.hid_seqs[sent_ind][word_ind]
            pos = state.g
            token = ev_seqs[sent_ind][word_ind]
            if pos == pos1: # reassign all indices with pos1 to pos0
                state.g = pos0
                new_models.pos.count(state.b[0], pos0, 1)
                new_models.pos.count(state.b[0], pos1, -1)
                new_models.lex.count(pos0, token, 1)
                new_models.lex.count(pos1, token, -1)
        from uhhmm import remove_pos_from_models, remove_pos_from_hid_seqs
        # sum the beta parameters of the POS tags being merged:
        new_models.pos.beta[pos0] = new_models.pos.beta[pos0] + new_models.pos.beta[pos1]
        remove_pos_from_models(new_models, pos1)
        remove_pos_from_hid_seqs(new_sample.hid_seqs, pos1)
        if new_models.pos.dist.shape[1] == 3:
            logging.warn("POS now down to only 3 (1) states")
        logging.debug("During merge the new shape of root is %s and exp is %s" % (str(new_models.root[0].dist.shape), str(new_models.exp[0].dist.shape) ) )
        assert new_models.pos.dist.shape[1] == new_models.root[0].dist.shape[1], "new pos dist size inconsistent with root: %d" % new_models.pos.dist.shape[1]  

    # acceptance probability calculation (see sections 4.3 and 2.1 in the Dahl paper)
    logging.debug("norm_const_lex0 = " + str(norm_const_lex[0]))
    logging.debug("norm_const_lex1 = " + str(norm_const_lex[1]))
    norm_const_pos_prior = norm_const(np.zeros((2)), alpha_pos)
    norm_const_pos = 0
    for awa in range(num_awa):
        norm_const_pos += norm_const(pos_counts[awa], alpha_pos) - norm_const_pos_prior
    logging.debug("norm_const_pos = " + str(norm_const_pos))
    norm_const_lex_merge = norm_const(np.sum(lex_counts, 0), alpha_lex) + norm_const(np.zeros((np.shape(lex_counts)[1])), alpha_lex)
    logging.debug("norm_const_lex_merge = " + str(norm_const_lex_merge))
    logging.debug("logprob_split = " + str(logprob_split))
    split_log_acc_ratio = norm_const_lex[0] + norm_const_lex[1] + norm_const_pos - logprob_split - norm_const_lex_merge
    log_acc_ratio = split_log_acc_ratio if split else -split_log_acc_ratio
    logging.debug("Log acceptance ratio = %s" % str(log_acc_ratio))
    if np.log(np.random.uniform()) < log_acc_ratio:
        logging.info("%s proposal was accepted." % ("Split" if split else "Merge") )
        new_sample.models = new_models  
        # for unit tests:
        # logging.debug("hid_seqs before split-merge:")
        # print_hid_seqs(sample.hid_seqs)
        # logging.debug("hid_seqs after split-merge:")
        # print_hid_seqs(new_sample.hid_seqs) 
        return new_models, new_sample #, [ind0, ind1]
    else:
        logging.info("%s proposal was rejected." % ("Split" if split else "Merge") )
        return models, sample #, None
