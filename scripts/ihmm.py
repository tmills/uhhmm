#!/usr/bin/env python3.4

import random
import logging
import time
import numpy as np
import ihmm_sampler as sampler
import pdb
import sys
from multiprocessing import Process,Queue,JoinableQueue

# The set of random variable values at one word
# There will be one of these for every word in the training set
class State:
    def __init__(self, state=None):
        if state == None:
            self.f = 0
            self.j = 0
            self.a = 0
            self.b = 0
            self.g = 0
        else:
            (self.f, self.j, self.a, self.b, self.g) = state

    def str(self):
        string = ''
        f_str = '+/' if self.f==1 else '-/'        
        string += f_str
        j_str = '+ ' if self.j==1 else '- '
        string += j_str
        
        string += str(self.a) + '/' + str(self.b) + ':' + str(self.g)
        
        return string

    def to_list(self):
        return (self.f, self.j, self.a, self.b, self.g)

# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0

# Historgam of how many instances of each state you have random sampled
# May be a field in Sample
class Stats:
    def __init__(self):
        ## number of each type of variable:
        self.numA = 0
        self.numB = 0
        self.numG = 0
        ## alpha is concentration parameter to dirichlet process/pitman yor model
        self.alpha0 = 0
        self.gamma = 0
        self.vi = 0

# A mapping from input space to output space. The Model class can be
# used to store counts during inference, and then know how to resample themselves
# if given a base distribution.
class Model:
    def __init__(self, shape):
        self.condCounts = np.zeros((shape[0],1), dtype=np.uint)
        self.pairCounts = np.zeros(shape, dtype=np.uint)
        self.dist = None

    def count(self, cond, out):
        self.condCounts[cond] += 1
        self.pairCounts[cond, out] += 1

    def dec(self, cond, out):
        self.condCounts[cond] -= 1
        self.pairCounts[cond,out] -= 1

    def sampleDirichlet(self, base):
        self.dist = sampler.sampleDirichlet(self, base, nullState=True)
        self.condCounts[:] = 0
        self.pairCounts[:] = 0

    def sampleBernoulli(self, base):
        self.dist = sampler.sampleDirichlet(self, base, nullState=False)
        self.condCounts[:] = 0
        self.pairCounts[:] = 0

# This class is not currently used. Could someday be used to resample
# all models if we give Model s more information about themselves.
class Models(list):
    def resample_all():
        for model in self:
            model.dist = sampleDirichlet(model)

# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
def sample_beam(ev_seqs, params, report_function):    
    
    global start_a, start_b, start_g 
    start_a = int(params.get('starta'))
    start_b = int(params.get('startb'))
    start_g = int(params.get('startg'))

    logging.basicConfig(level=logging.WARN)
    logging.info("Starting beam sampling")
    
    burnin = int(params.get('burnin'))
    iters = int(params.get('sample_iters'))
    num_samples = int(params.get('num_samples'))
    num_procs = int(params.get('num_procs'))
    debug = params.get('debug')
    
#    num_iters = burnin + (samples-1)*iters
    samples = []
    
    maxLen = max(map(len, ev_seqs))
    
    sample = Sample()
    sample.alpha_a = float(params.get('alphaa'))
    sample.alpha_b = float(params.get('alphab'))
    sample.alpha_g = float(params.get('alphag'))
    sample.alpha_f = float(params.get('alphaf'))
    sample.alpha_j = float(params.get('alphaj'))
    sample.beta_a = np.ones((1,start_a+1)) / start_a
    sample.beta_a[0][0] = 0
    sample.beta_b = np.ones((1,start_b+1)) / start_b
    sample.beta_b[0][0] = 0
    sample.beta_g = np.ones((1,start_g+1)) / start_g
    sample.beta_g[0][0] = 0
    sample.beta_f = np.ones((1,2)) / 2
    sample.beta_j = np.ones((1,2)) / 2
    sample.gamma = float(params.get('gamma'))
    sample.discount = float(params.get('discount'))
    
    
    models = Models()
    
    logging.info("Initializing state")    
    hid_seqs = initialize_state(ev_seqs, models)
    sample.hid_seqs = hid_seqs
    ## Sample distributions for all the model params and emissions params
    ## TODO -- make the Models class do this in a resample_all() method
    models.lex.sampleDirichlet(params['h'])
    models.pos.sampleDirichlet(sample.alpha_g * sample.beta_g)
    models.start.sampleDirichlet(sample.alpha_b * sample.beta_b)
    models.cont.sampleDirichlet(sample.alpha_b * sample.beta_b)
    models.act.sampleDirichlet(sample.alpha_a * sample.beta_a)
    models.root.sampleDirichlet(sample.alpha_a * sample.beta_a)
    models.reduce.sampleBernoulli(sample.alpha_j * sample.beta_j)
    models.fork.sampleBernoulli(sample.alpha_f * sample.beta_f)
    
    stats = Stats()
    
    logging.debug(ev_seqs[0])
    logging.debug(list(map(lambda x: x.str(), hid_seqs[0])))
    
    iter = 0
    
    while len(samples) < num_samples:
             
        ## These values keep track of actual maxes not user-specified --
        ## so if user specifies 10 to start this will be 11 because of state 0 (init)
        a_max = models.act.dist.shape[1]
        b_max = models.cont.dist.shape[1]
        g_max = models.pos.dist.shape[1]
                
        ## How many total states are there?
        ## 2*2*|Act|*|Awa|*|G|
        totalK = 2 * 2 * a_max * b_max * g_max
        inf_procs = dict()
        cur_proc = 0

        sent_q = JoinableQueue() ## Input queue
        state_q = Queue() ## Output queue
        
            
        ## Place all sentences into the input queue
        for sent_index,sent in enumerate(ev_seqs):
            sent_q.put((sent_index,sent))
            
        ## Initialize all the sub-processes with their input-output queues,
        ## read-only models, and dimensions of matrix they'll need
        t0 = time.time()
        for cur_proc in range(0,num_procs):
            ## For each sub-process add a "None" element to the queue that tells it that
            ## we are out of sentences (we've added them all above)
            sent_q.put(None)
            
            ## Initialize and start the sub-process
            inf_procs[cur_proc] = Sampler(sent_q, state_q, models, totalK, maxLen+1, cur_proc)
            if debug:
                ## calling run instead of start just treats it like a plain object --
                ## doesn't actually do a fork. So we'll just block here for it to finish
                ## rather than needing to join later.
                ## Then we can use pdb() for debugging inside the thread.
                inf_procs[cur_proc].run()
            else:
                inf_procs[cur_proc].start()

        ## Close the queue
        sent_q.join()
        t1 = time.time()
        logging.info(".")
        logging.debug("Sampling time for this batch is %d s" % (t1-t0))
        
        t0 = time.time()
        num_processed = 0
        sample_map = dict()
        while not state_q.empty():
            num_processed += 1
            (sent_index, sent_sample, log_prob) = state_q.get()
            #logging.debug("Incrementing count for sent index %d and %d sentences left in queue" % (sent_index, len(ev_seqs)-num_processed))
            if sent_index % 10 == 0:
                logging.info("Processed sentence {0}".format(sent_index))
            #pdb.set_trace()
            increment_counts(sent_sample, ev_seqs[sent_index], models)
            sample_map[sent_index] = sent_sample
            sample.log_prob += log_prob

        ## samples got unsorted by queueing them so resort them just for the purpose 
        ## of debugging.
        for key in sorted(sample_map.keys()):
            sample.hid_seqs.append(sample_map[key])

        t1 = time.time()
        logging.debug("Building counts tables took %d s" % (t1-t0))
        
        if iter >= burnin and (iter-burnin) % iters == 0:
            samples.append(sample)
            report_function(sample)
            logging.info(".\n")
        
        t0 = time.time()

        next_sample = Sample()
        ## TODO Sample hyper-parameters
        ## This is, e.g., where we might add categories to the a,b,g variables with
        ## stick-breaking. Without that, the values will stay what they were 
        next_sample.alpha_f = sample.alpha_f
        next_sample.beta_f = sample.beta_f
        next_sample.alpha_j = sample.alpha_j
        next_sample.beta_j = sample.beta_j
        next_sample.alpha_a = sample.alpha_a
        next_sample.beta_a = sample.beta_a
        next_sample.alpha_b = sample.alpha_b
        next_sample.beta_b = sample.beta_b
        next_sample.alpha_g = sample.alpha_g
        next_sample.beta_g = sample.beta_g
        sample = next_sample
                
        ## Sample distributions for all the model params and emissions params
        ## TODO -- make the Models class do this in a resample_all() method
        models.lex.sampleDirichlet(params['h'])
        models.pos.sampleDirichlet(sample.alpha_g * sample.beta_g)
        models.start.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.cont.sampleDirichlet(sample.alpha_b * sample.beta_b)
        models.act.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.root.sampleDirichlet(sample.alpha_a * sample.beta_a)
        models.reduce.sampleBernoulli(sample.alpha_j * sample.beta_j)
        models.fork.sampleBernoulli(sample.alpha_f * sample.beta_f)
        t1 = time.time()
        
        logging.debug("Resampling models took %d s" % (t1-t0))
        sample.models = models
        sample.iter = iter
        
        iter += 1
        

    return (samples, stats)

# This class does the actual sampling. It is a Python process rather than a Thread
# because python threads do not work well due to the global interpreter lock (GIL), 
# which only allows one thread at a time to access the interpreter. Making it a process
# is requires more indirect communicatino using shared input/output queues between 
# different sampler instances
class Sampler(Process):
    def __init__(self, in_q, out_q, models, totalK, maxLen, tid):
        Process.__init__(self)
        self.in_q = in_q
        self.out_q = out_q
        self.models = models
        self.K = totalK
        self.dyn_prog = np.zeros((totalK,maxLen))
        self.tid = tid
    
    def set_data(self, sent):
        self.sent = sent

    def run(self):
        self.dyn_prog[:,:] = -np.inf
        #logging.debug("Starting forward pass in thread %s", self.tid)

        while True:
            task = self.in_q.get()
            if task == None:
                self.in_q.task_done()
                break
            
            (sent_index, sent) = task
            t0 = time.time()
            self.dyn_prog[:,:] = -np.inf
            (self.dyn_prog, log_prob) = self.forward_pass(self.dyn_prog, sent, self.models, self.K, sent_index)
            sent_sample = self.reverse_sample(self.dyn_prog, sent, self.models, self.K, sent_index)
            t1 = time.time()
            self.in_q.task_done()
            self.out_q.put((sent_index, sent_sample,log_prob))
            
            if log_prob > 0:
                logging.error("Sentence %d had positive log probability %f" % (sent_index, log_prob))
            
            #logging.debug("Thread %d required %d s to process sentence.", self.tid, (t1-t0))

    def get_sample(self):
        return self.sent_sample

    def forward_pass(self,dyn_prog,sent,models,totalK, sent_index):
        ## keep track of forward probs for this sentence:
        for index,token in enumerate(sent):
            if index == 0:
                g0_ind = getStateIndex(0,0,0,0,0)
                dyn_prog[g0_ind:g0_ind+g_max,0] = models.lex.dist[:,token]
                logging.debug(dyn_prog[g0_ind:g0_ind+g_max,0])
            else:
                for prevInd in range(0,dyn_prog.shape[0]):
                    if dyn_prog[prevInd,index-1] == -np.inf:
                        continue

                    (prevF, prevJ, prevA, prevB, prevG) = extractStates(prevInd, totalK)

                    assert index == 1 or (prevA != 0 and prevB != 0 and prevG != 0), 'Unexpected values in sentence {0} with non-zero probabilities: {1}, {2}, {3} at index {4}, and f={5} and j={6}, ind={7}'.format(sent_index,prevA, prevB, prevG, index, prevF, prevJ, prevInd)
                
                    cumProbs = np.zeros((5,1))
                    prevBG = bg_state(prevB,prevG)
                
                    ## Sample f & j:
                    for f in (0,1):
                        if index == 1 and f == 0:
                            continue

                        cumProbs[0] = dyn_prog[prevInd,index-1] + models.fork.dist[prevBG,f]
                        if index == 1:
                            j = 0
                        else:
                            j = f
                        
                        ## At depth 1 -- no probability model for j 
                        cumProbs[1] = cumProbs[0]
                        
                        
                        for a in range(1,a_max):
                            if f == 0 and j == 0:
                                ## active transition:
                                cumProbs[2] = cumProbs[1] + models.act.dist[prevA,a]
                            elif f == 1 and j == 0:
                                ## root -- technically depends on prevA and prevG
                                ## but in depth 1 this case only comes up at start
                                ## of sentence and prevA will always be 0
                                cumProbs[2] = cumProbs[1] + models.root.dist[prevG,a]
                            elif f == 1 and j == 1 and prevA == a:
                                cumProbs[2] = cumProbs[1]
                            else:
                                ## zero probability here
                                continue
                        
                            prevAa = aa_state(prevA, a)
                        
                            for b in range(1,b_max):
                                if j == 1:
                                    cumProbs[3] = cumProbs[2] + models.cont.dist[prevBG,b]
                                else:
                                    cumProbs[3] = cumProbs[2] + models.start.dist[prevAa,b]
                            
                                logging.debug(cumProbs)
                                # Multiply all the g's in one pass:
                                ## range gets the range of indices in the forward pass
                                ## that are contiguous in the state space
                                state_range = getStateRange(f,j,a,b)
                                
                                logging.debug(dyn_prog[state_range, index])
                                
                                range_probs = cumProbs[3] + models.pos.dist[b,:] + models.lex.dist[:,token]
                                logging.debug(range_probs)
                                
                                dyn_prog[state_range,index] = log_vector_add(dyn_prog[state_range,index], range_probs)

        ## For the last token, multiply in the probability
        ## of transitioning to the end state. also can add up
        ## total probability of data given model here.
        sentence_log_prob = -np.inf
        last_index = len(sent)-1
        for state in range(0,dyn_prog.shape[0]):
            (f,j,a,b,g) = extractStates(state, totalK)
            curBG = bg_state(b,g)
            dyn_prog[state,last_index] += ((models.fork.dist[curBG,0] + models.reduce.dist[a,1]))
            sentence_log_prob = log_add(sentence_log_prob, dyn_prog[state, last_index])
            logging.debug(dyn_prog[state,last_index])
                       
            if (a == 0 or b == 0 or g == 0) and dyn_prog[state, last_index] != -np.inf:
                logging.error("Error: Non-zero probability at g=0 in forward pass!")
                sys.exit(-1)

        return dyn_prog, sentence_log_prob

    def reverse_sample(self, dyn_prog, sent, models, totalK, sent_index):            
        sample_seq = []
        sample_log_prob = 0
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        dyn_prog[:,last_index] = normalize_from_log(dyn_prog[:,last_index])
        sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,last_index]))
                
        sample_seq.append(State(extractStates(sample_t, totalK)))
        if sample_seq[-1].a == 0 or sample_seq[-1].b == 0 or sample_seq[-1].g == 0:
            logging.error("Error: First sample has a|b|g = 0")
            sys.exit(-1)
  
        for t in range(len(sent)-2,-1,-1):
            for ind in range(0,dyn_prog.shape[0]):
                if dyn_prog[ind,t] == -np.inf:
                    continue

                (pf,pj,pa,pb,pg) = extractStates(ind,totalK)
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                prevBG = bg_state(pb,pg)
                trans_prob = models.fork.dist[prevBG,nf]
                if nf == 0:
                    trans_prob += models.reduce.dist[pa,nj]
      
                if nf == 0 and nj == 0:
                    trans_prob += models.act.dist[pa,na]
                elif nf == 1 and nj == 0:
                    trans_prob += models.root.dist[pg,na]
                elif nf == 1 and nj == 1:
                    if na != pa:
                        trans_prob = -np.inf
      
                if nj == 0:
                    prevAA = aa_state(pa,na)
                    trans_prob += models.start.dist[prevAA,nb]
                else:
                    trans_prob += models.cont.dist[prevBG,nb]
      
                trans_prob += models.pos.dist[nb,ng]
                dyn_prog[ind,t] += trans_prob

            dyn_prog[:,t] = normalize_from_log(dyn_prog[:,t])
            sample_t = sum(np.random.random() > np.cumsum(dyn_prog[:,t]))
            state_list = extractStates(sample_t, totalK)
            
            sample_state = State(state_list)
            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
            sample_seq.append(sample_state)

        sample_seq.reverse()
        logging.info("Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        return sample_seq

# Randomly initialize all the values for the hidden variables in the 
# sequence. Obeys constraints (e.g., when f=1,j=1 a=a_{t-1}) but otherwise
# samples randomly.
def initialize_state(ev_seqs, models):
    global a_max, b_max, g_max
    a_max = start_a+1
    b_max = start_b+1
    g_max = start_g+1
    
    ## One fork model:
    models.fork = Model(((g_max)*(b_max), 2))
    ## Two join models:
#    models.trans = Model((g_max*b_max, 2))
    models.reduce = Model((a_max, 2))
    ## One active model:
    models.act = Model((a_max, a_max))
    models.root = Model((g_max, a_max))
    ## two awaited models:
    models.cont = Model(((g_max)*(b_max),b_max))
    models.start = Model(((a_max)*(a_max), b_max))
    ## one pos model:
    models.pos = Model((b_max, g_max))
    ## one lex model:
    models.lex = Model((g_max, max(map(max,ev_seqs))+1))
    
    logging.debug("Value of amax=%d, b_max=%d, g_max=%d", a_max, b_max, g_max)
    
    state_seqs = list()
    for sent in ev_seqs:
        hid_seq = list()
        for index,word in enumerate(sent):
            state = State()
            ## special case for first word
            if index == 0:
                state.f = 0
                state.j = 0
                state.a = 0
                state.b = 0
            else:
                if index == 1:
                    state.f = 1
                    state.j = 0
                else:
                    if random.random() > 0.5:
                        state.f = 1
                    else:
                        state.f = 0
                    ## j is deterministic in the middle of the sentence
                    state.j = state.f
                    
                if state.f == 1 and state.j == 1:
                    state.a = prev_state.a
                else:
                    state.a = np.random.randint(1,a_max)

                state.b = np.random.randint(1,b_max)

            state.g = np.random.randint(1,g_max)
                    
            prev_state = state  
                            
            hid_seq.append(state)
            
        increment_counts(hid_seq, sent, models)
        state_seqs.append(hid_seq)

    return state_seqs

def increment_counts(hid_seq, sent, models):
    ## for every state transition in the sentence increment the count
    ## for the condition and for the output
    for index,word in enumerate(sent):
        state = hid_seq[index]
        if index != 0:
            prevBG = bg_state(prevState.b, prevState.g)
            ## Count F & J
            if index == 1:
                models.root.count(prevState.g, state.a)
            else:
                models.fork.count(prevBG, state.f)

                ## Count A & B
                if state.f == 0 and state.j == 0:
                    models.act.count(prevState.a, state.a)

            if state.f == 0 and state.j == 0:
                models.reduce.count(prevState.a, state.j)
                                
            if state.j == 0:
                models.start.count(aa_state(prevState.a, state.a), state.b)
            else:
                models.cont.count(prevBG, state.b)

            
        ## Count G
        models.pos.count(state.b, state.g)
        
        ## Count w
        models.lex.count(state.g, word)
        
        prevState = state
    
    prevBG = bg_state(hid_seq[-1].b, hid_seq[-1].g)
    models.fork.count(prevBG, 0)
    models.reduce.count(hid_seq[-1].a, 1)

## Don't think we actually need this since we are not using counts to integrate
## out models -- instead we sample models so we can just reset all counts to 0
## after resampling models at each iteration
def decrement_counts(hid_seq, sent, models):
    ## for every state transition in the sentence decrement the count
    ## for the condition and and the generated value
    for index,word in enumerate(sent):
        state = hid_seq[index]
        if index != 0:
            prevBG = bg_state(prevState.b, prevState.g)
            if index == 1:
                models.root.dec(prevState.g, state.a)
            else:
                
                models.fork.dec(prevBG, state.f)
                
                if state.f == 0 and state.j == 0:
                    models.act.dec(prevState.a, state.a)
                
            if state.j == 0:
                models.start.dec(aa_state(prevState.a, state.a), state.b)
            else:
                models.cont.dec(prevBG, state.b)
                
        models.pos.dec(state.b, state.g)
        models.lex.dec(state.g, sent[index])
        
        prevState = state

def log_vector_add(x, y):
    p = np.zeros(x.shape)
    for ind,val in enumerate(x):
        p[ind] = log_add(x[ind],y[ind])
    
    return p
    
## From here: https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
def log_add(x, y):
    ## Step 0: An addition -- if one of the values is 0 that means it has not been initialized yet
    if x == 0:
        return y
    
    if y == 0:
        return x
    
    if x == -np.inf:
        return y
        
    if y == -np.inf:
        return x
    
    ## Step 1: make x bigger
    if y > x:
        temp = y
        y = x
        x = temp
    
    ## Step 2:
    if x == sys.float_info[3]:   ## min value
        return x
        
    ## Step 3: How far down is y from x
    ## If it's small, ignore:
    neg_diff = y - x
    if neg_diff < -20:
        return x
    
    ## Step 4: Otherwise use algebra:
    return x + np.log10(1.0 + 10**neg_diff)

def normalize_from_log(log_dist):
    ## scale back up to a safe range
    log_dist -= log_dist.max()
    
    ## Back into prob space
    dist = 10**log_dist
    
    dist /= dist.sum()
    
    return dist

def bg_state(b, g):
    global g_max
    return b*g_max + g

def aa_state(prevA, a):
    global a_max
    return prevA * a_max + a
    
def extractStates(index, totalK):
    global a_max, b_max, g_max
    
    ## First -- we only care about a,b,g for computing next state so factor
    ## out the a,b,g
    f_ind = 0
    f_split = totalK / 2
    if index > f_split:
        index = index - f_split
        f_ind = 1
    
    j_ind = 0
    j_split = f_split / 2
    if index > j_split:
        index = index - j_split
        j_ind = 1
    
    g_ind = index % g_max
    index = (index-g_ind) / g_max
    
    b_ind = index % b_max
    index = (index-b_ind) / b_max
    
    a_ind = index % a_max
    
    ## Make sure all returned values are ints:
    return map(int, (f_ind,j_ind,a_ind, b_ind, g_ind))

def getStateIndex(f,j,a,b,g):
    global a_max, b_max, g_max
    return (((f*2 + j)*a_max + a) * b_max + b)*g_max + g

def getStateRange(f,j,a,b):
    global g_max
    start = getStateIndex(f,j,a,b,0)
    return range(start,start+g_max)
