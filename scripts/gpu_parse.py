import logging
import indexer

def parse(start_ind, end_ind, distributer, ev_seqs, hid_seqs):
    distributer.submitSentenceJobs(start_ind, end_ind)
    num_processed = 0
    parses = distributer.get_parses()
    logprobs = 0
    state_list = []
    state_indices = []
    for parse in parses:
        num_processed += 1
        # logging.info(''.join([x.str() for x in parse.state_list]))
        # logging.info(parse.success)
        if parse.success:
            try:
                state_list.append(parse.state_list)
                state_indices.append(ev_seqs[parse.index])
                # logging.info('The state sequence is ' + ' '.join([str(indexer.getStateIndex(x.j, x.a, x.b, x.f, x.g)) for x in parse.state_list]))
                # logging.info(' '.join([x.str() for x in parse.state_list]))

                # increment_counts(parse.state_list, ev_seqs[parse.index], models)

                # logging.info('Good parse:')
                # logging.info(' '.join([x.str() for x in parse.state_list]))
                # logging.info('The index is %d' % parse.index)
            except:
                logging.error('This parse is bad:')
                logging.error('The sentence is ' + ' '.join([str(x) for x in ev_seqs[parse.index]]))
                logging.error('The state sequence is ' + ' '.join(
                    [str(indexer.getStateIndex(x.j, x.a, x.b, x.f, x.g)) for x in parse.state_list]))
                logging.error(' '.join([x.str() for x in parse.state_list]))
                logging.error('The index is %d' % parse.index)
                raise
            logprobs += parse.log_prob
        hid_seqs[parse.index] = parse.state_list
    return logprobs