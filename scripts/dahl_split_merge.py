#!/usr/bin/env python3.4
import numpy as np
import logging
from scipy.special import gammaln

# sentence and word indices at given position in sequence
def indices (cum_length, ind):
    sent_ind = np.where(cum_length > ind)[0][0]
    word_ind = ind if sent_ind == 0 else ind - cum_length[sent_ind-1]
    return sent_ind, int(word_ind)

# log emission posterior normalizing constant given word counts in given state
# assumes multinomial-dirichlet model
def norm_const(obs_counts, alpha=.01):
    return sum(gammaln(obs_counts+alpha)) - gammaln(sum(obs_counts+alpha))
    
# log probability of HMM state partition 
# assumes multinomial-dirichlet model for transition matrix rows
# assumes pos tags are consecutive integers starting at 0
def trans_term (sample, nstates, alpha=1):
    counts = np.zeros((nstates, nstates))
    for seq in sample.hid_seqs:
        for word_index in range(len(seq)-1):
          counts[seq[word_index].g, seq[word_index+1].g] += 1
        
    term = 0
    for state in range(nstates):
        term += gammaln(nstates*alpha) - gammaln(nstates*alpha + np.sum(counts, 1)[state])
        for state1 in range(nstates):
          term += gammaln(alpha + counts[state, state1]) - gammaln(alpha)

    return term

def perform_split_merge_operation(models, sample, ev_seqs, params, iter):
    num_tokens = sum(map(len, ev_seqs))
    cum_length = np.cumsum(list(map(len, ev_seqs)))

    ## Need to copy the models otherwise we'll just have another pointer
    new_models = models.copy()
    #new_models.exp[0].beta=1
    new_sample = sample

    ## Split-merge for pos tags:            
    ind0 = np.random.randint(num_tokens) # anchor 0
    ind1 = np.random.randint(num_tokens) # anchor 1
    sent0_ind, word0_ind = indices(cum_length, ind0)
    sent1_ind, word1_ind = indices(cum_length, ind1)
    pos0 = sample.hid_seqs[sent0_ind][word0_ind].g
    pos1 = sample.hid_seqs[sent1_ind][word1_ind].g
    
    split = (pos0 == pos1) # whether to split or merge
    
    # indices with state pos0 or pos1
    subset = np.array([]) 
    for ind in range(num_tokens):
        sent_ind, word_ind = indices(cum_length, ind)
        pos = sample.hid_seqs[sent_ind][word_ind].g
        if ((pos == pos0) | (pos == pos1)):
            subset = np.append(subset, ind)

    np.random.shuffle(subset)

    # indices with state pos0 or pos1, excluding anchors
    subset1 = np.delete(subset, np.where(subset == ind0)[0])
    subset1 = np.delete(subset1, np.where(subset1 == ind1)[0])
    
    sets = [[ind0], [ind1]]
    logprob_prop = 0
    if not split:
        st = {pos0: 0, pos1: 1}
   
    unique_obs = set(sum(ev_seqs, []))
    obs_counts = np.zeros((2, len(unique_obs)))
    obs_counts[0, ev_seqs[sent0_ind][word0_ind]-1] = 1
    obs_counts[1, ev_seqs[sent1_ind][word1_ind]-1] = 1
    norm_consts = np.array([norm_const(obs_counts[0]), norm_const(obs_counts[1])])            
    
    for ind in subset1: 
        sent_ind, word_ind = indices(cum_length, ind)
        newcounts = np.copy(obs_counts)
        newcounts[:, ev_seqs[sent_ind][word_ind]-1] += 1
        logterms = np.array([norm_const(newcounts[0]), norm_const(newcounts[1])]) - norm_consts
        terms = [np.exp(logterms[0] - max(logterms)) * len(sets[0]), 
                 np.exp(logterms[1] - max(logterms)) * len(sets[1])]
        newstate = np.random.binomial(1, terms[1]/sum(terms)) if split \
          else st[sample.hid_seqs[sent_ind][word_ind].g]
        logprob_prop += logterms[newstate] - max(logterms) + \
          np.log(len(sets[newstate])) - np.log(sum(terms))
        sets[newstate].append(ind)
        norm_consts[newstate] += logterms[newstate]
        obs_counts[newstate] = newcounts[newstate]
    
    nstates = models.pos.dist.shape[1]-1
    tt = trans_term(sample, nstates)
    if split:
        logging.info("Performing split operation of pos tag %d at iteration %d" % (pos0,iter))
        from uhhmm import break_g_stick
        break_g_stick(new_models, new_sample, params) # add new pos tag variable
        for ind in sets[1]:
            sent_ind, word_ind = indices(cum_length, ind)
            state = new_sample.hid_seqs[sent_ind][word_ind]
            state.g = nstates
            new_models.pos.pairCounts[state.b[0]][pos0] -= 1
            new_models.pos.pairCounts[state.b[0]][state.g] += 1
        tt = trans_term(new_sample, nstates+1) - tt
    else:
        (pos0, pos1) = (min(pos0, pos1), max(pos0, pos1))
        pos_last = nstates-1
        logging.info("Performing merge operation between pos tags " + \
          "%d and %d at iteration %d" % (pos0, pos1, iter))
        if models.pos.dist.shape[1] == 3:
            logging.warn("Performing a merge with only 1 state left")              
        for ind in range(num_tokens):
            sent_ind, word_ind = indices(cum_length, ind)
            state = new_sample.hid_seqs[sent_ind][word_ind]
            pos = state.g
            if pos == pos1:
                state.g = pos0
                new_models.pos.pairCounts[state.b[0]][pos0] += 1
                new_models.pos.pairCounts[state.b[0]][pos1] -= 1
            elif pos == pos_last: 
                new_sample.hid_seqs[sent_ind][word_ind].g = pos1
                new_models.pos.pairCounts[state.b[0]][pos1] += 1
                new_models.pos.pairCounts[state.b[0]][pos_last] -= 1

        from uhhmm import remove_pos_from_models
        remove_pos_from_models(new_models, pos1)
        if new_models.pos.dist.shape[1] == 3:
            logging.warn("POS now down to only 3 (1) states")
            
        tt -= trans_term(new_sample, nstates-1) 

    split_logprob_acc = norm_consts[0] + norm_consts[1] + tt - logprob_prop - \
      norm_const(np.sum(obs_counts, 0)) - norm_const(np.zeros((np.shape(obs_counts)[1])))

    if (split and (np.log(np.random.uniform()) < split_logprob_acc)) or \
       ((not split) and (np.log(np.random.uniform()) < - split_logprob_acc)):
        models = new_models
        sample = new_sample
        logging.info("%s proposal was accepted." % ("Split" if split else "Merge") )
    else:
        logging.info("%s proposal was rejected." % ("Split" if split else "Merge") )

