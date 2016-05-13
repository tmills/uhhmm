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
# assumes multinomial-dirichlet model
# used for lex | pos and pos | awa
def norm_const(obs_counts, alpha=.2):
    return sum(gammaln(obs_counts+alpha)) - gammaln(sum(obs_counts+alpha))

# Sequentially Allocated Merge Split algorithm (Dahl 2005, section 4.1)
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.131.3712&rep=rep1&type=pdf    
def perform_split_merge_operation(models, sample, ev_seqs, params, iter, equal=True):
    num_tokens = sum(map(len, ev_seqs))
    cum_length = np.cumsum(list(map(len, ev_seqs)))
    ## Need to copy the models otherwise we'll just have another pointer
    new_models = copy.deepcopy(models)
    new_sample = sample

    ## Split-merge for pos tags:            
    ind0 = np.random.randint(num_tokens) # anchor 0
    sent0_ind, word0_ind = indices(cum_length, ind0)
    pos0 = sample.hid_seqs[sent0_ind][word0_ind].g
    if equal: 
        split = (np.random.binomial(1, .5) > .5)
        subset = np.array([]) 
        for ind in range(num_tokens):
            sent_ind, word_ind = indices(cum_length, ind)
            state = sample.hid_seqs[sent_ind][word_ind]
            if (((state.g == pos0) and split) or ((state.g != pos0) and not split)):
                subset = np.append(subset, ind)
        ind1 = np.random.choice(subset)
    else:
        ind1 = np.random.randint(num_tokens) # anchor 1

    sent1_ind, word1_ind = indices(cum_length, ind1)
    pos1 = sample.hid_seqs[sent1_ind][word1_ind].g
    split = (pos0 == pos1) # whether to split or merge

    alpha_lex = params['h'][0,1:]
    logging.debug("Lexical Dirichlet pseudocounts:")
    logging.debug(alpha_lex)
    alpha_pos_vec = models.pos.alpha * models.pos.beta[1:]
    alpha_pos = [alpha_pos_vec[pos0-1], alpha_pos_vec[pos1-1]]
    logging.debug("POS Dirichlet pseudocounts:")
    logging.debug(alpha_pos)
    
    # indices with state pos0 or pos1
    subset = np.array([]) 
    nstates = models.pos.dist.shape[1]-2
    nawa = models.pos.dist.shape[0]
    for ind in range(num_tokens):
        sent_ind, word_ind = indices(cum_length, ind)
        state = sample.hid_seqs[sent_ind][word_ind]
        if ((state.g == pos0) or (state.g == pos1)):
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
    awa0 = sample.hid_seqs[sent0_ind][word0_ind].b[0]
    awa1 = sample.hid_seqs[sent1_ind][word1_ind].b[0]
    pos_counts = np.zeros((nawa, 2))
    pos_counts[awa0, 0] += 1
    pos_counts[awa1, 1] += 1
    norm_consts = np.array([norm_const(obs_counts[0], alpha_lex), norm_const(obs_counts[1], alpha_lex)])
    for ind in subset1: 
        sent_ind, word_ind = indices(cum_length, ind)
        newcounts = np.copy(obs_counts)
        newcounts[:, ev_seqs[sent_ind][word_ind]-1] += 1
        awa = sample.hid_seqs[sent_ind][word_ind].b[0]
        newcounts_pos0 = np.copy(pos_counts[awa])
        newcounts_pos1 = np.copy(pos_counts[awa])
        newcounts_pos0[0] += 1
        newcounts_pos1[1] += 1
        logterms1 = np.array([norm_const(newcounts[0], alpha_lex), norm_const(newcounts[1], alpha_lex)]) - norm_consts
        logterms = logterms1 + np.array([norm_const(newcounts_pos0, alpha_pos), norm_const(newcounts_pos1, alpha_pos)])
        terms = [np.exp(logterms[0] - max(logterms)) * len(sets[0]), 
                 np.exp(logterms[1] - max(logterms)) * len(sets[1])]
        newstate = np.random.binomial(1, terms[1]/sum(terms)) if split else st[sample.hid_seqs[sent_ind][word_ind].g]
        logprob_prop += logterms[newstate] - max(logterms) + np.log(len(sets[newstate])) - np.log(sum(terms))
        sets[newstate].append(ind)
        norm_consts[newstate] += logterms1[newstate]
        obs_counts[newstate] = newcounts[newstate]
        pos_counts[awa] = newcounts_pos1 if (newstate == 1) else newcounts_pos0
    
    logging.debug("During split-merge the shape of root is %s and exp is %s" % (str(models.root[0].dist.shape), str(models.exp[0].dist.shape) ) )
    if split:
        logging.info("Performing split operation of pos tag %d at iteration %d" % (pos0,iter))
        from uhhmm import break_g_stick
        break_g_stick(new_models, new_sample, params) # add new pos tag variable
        for ind in sets[1]:
            sent_ind, word_ind = indices(cum_length, ind)
            state = new_sample.hid_seqs[sent_ind][word_ind]
            token = ev_seqs[sent_ind][word_ind]
            state.g = nstates+1
            new_models.pos.count(state.b[0], pos0, -1)
            new_models.pos.count(state.b[0], state.g, 1)
            new_models.lex.count(pos0, token, -1)
            new_models.lex.count(state.g, token, 1)
        logging.debug("During split the new shape of root is %s and exp is %s" % (str(new_models.root[0].dist.shape), str(new_models.exp[0].dist.shape) ) )
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
            ## Reassign things from pos1 into pos0
            if pos == pos1:
                state.g = pos0
                new_models.pos.count(state.b[0], pos0, 1)
                new_models.pos.count(state.b[0], pos1, -1)
                new_models.lex.count(pos0, token, 1)
                new_models.lex.count(pos1, token, -1)
                continue
            ## Since we'll remove the last pos tag, move everything from that tag
            ## into the pos1 index.
            if pos == nstates: 
                state.g = pos1
                new_models.pos.count(state.b[0], pos1, 1)
                new_models.lex.count(pos1, token, 1)
        from uhhmm import remove_pos_from_models
        remove_pos_from_models(new_models, nstates)
        if new_models.pos.dist.shape[1] == 3:
            logging.warn("POS now down to only 3 (1) states")
        logging.debug("During merge the new shape of root is %s and exp is %s" % (str(new_models.root[0].dist.shape), str(new_models.exp[0].dist.shape) ) )

    # acceptance probability calculation follows sections 4.3 and 2.1
    logging.debug("norm_const0 = " + str(norm_consts[0]))
    logging.debug("norm_const1 = " + str(norm_consts[1]))
    norm_const_pos_prior = norm_const(np.zeros((2)), alpha_pos)
    norm_const_pos = 0
    for i in range(nawa):
        norm_const_pos += norm_const(pos_counts[awa], alpha_pos) - norm_const_pos_prior
    logging.debug("norm_const_pos = " + str(norm_const_pos))
    nc = norm_const(np.sum(obs_counts, 0), alpha_lex) + norm_const(np.zeros((np.shape(obs_counts)[1])), alpha_lex)
    logging.debug("norm_const_merge = " + str(nc))
    logging.debug("logprob_prop = " + str(logprob_prop))
    split_logprob_acc = norm_consts[0] + norm_consts[1] + norm_const_pos - logprob_prop - nc
    logprob_acc = split_logprob_acc if split else -split_logprob_acc
    logging.debug("Log acceptance probability = %s" % str(logprob_acc))
    if np.log(np.random.uniform()) < logprob_acc:
        logging.info("%s proposal was accepted." % ("Split" if split else "Merge") )
        new_sample.models = new_models
        return new_models, new_sample
    else:
        logging.info("%s proposal was rejected." % ("Split" if split else "Merge") )
        return models, sample

