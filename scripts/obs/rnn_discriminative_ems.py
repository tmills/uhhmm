import logging

import torch
from typing import List
import torch.optim
from .rnn_generative_ems import RNNEntry, RNNEntryList

NUM_LAYERS = 2
BIDIRECTIONAL = 1
NUM_ITERS = 5
class RNNDiscriminativeEntry(RNNEntry):
    def get_nll(self):
        return 0 - self.count @ torch.log(self.prob)

class RNNDiscriminativeEntryList(RNNEntryList):
    def set_scores(self, probs, w_logistic=None, b_logistic=None):
        total_nll = 0
        # print(probs)
        for entry_index, entry in enumerate(self.entries):
            entry.set_prob(probs[entry_index])
            total_nll += entry.get_nll()
        return total_nll

class RNNDiscriminativeEmission(torch.nn.Module):
    def __init__(self, abp_domain_size, vocab, hidden_size=50, num_layers=NUM_LAYERS,
                 training_iters=NUM_ITERS, use_cuda=True, bidirectional=BIDIRECTIONAL):
        super(RNNDiscriminativeEmission, self).__init__()
        self.bidirectional = bidirectional
        self.training_iters = training_iters
        self.use_cuda = use_cuda
        self.vocab = vocab
        # print(vocab)
        self.vocab_size = 0
        self.char_set = {}
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = 0
        self._generate_vocab()
        self.abp_domain_size = abp_domain_size # do not take into account ROOT
        self.input_size = len(self.char_set)
        self.total_word_counts = {}
        self.non_terms = {}
        self.rnn = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, bidirectional=self.bidirectional,
                                batch_first=True)
        self.final_layer = torch.nn.Linear(self.hidden_size * self.num_layers * (1+self.bidirectional),
                                           self.abp_domain_size)
        self.h_0 = torch.nn.Parameter(data=torch.zeros(
            self.num_layers * (1+self.bidirectional), 1, self.hidden_size))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        print(self)

    def _generate_vocab(self):
        chars = {}
        for word in self.vocab.values():
            char_list = list(word)
            for char in char_list:
                if char not in chars:
                    chars[char] = len(chars)
            if len(char_list) > self.max_len:
                self.max_len = len(char_list)
        self.char_set = chars
        self.vocab_size = len(self.char_set)
        # print(self.char_set)

    def update_distrbution(self, pcfg_counts):
        if self.use_cuda:
            self.cuda()
        self.total_word_counts = {}
        self._generate_input(pcfg_counts)
        self._rnn_step()
        self._update_pcfg(pcfg_counts)
        if self.use_cuda:
            self.cpu()
            self.entries = []

    def _generate_input(self, pcfg_counts):
        entries = []
        # print(pcfg_counts)
        for parent in pcfg_counts:
            self.non_terms[int(parent.symbol()) - 1] = parent  # no ROOT

        for word_index in self.vocab:
            word_int = str(word_index)
            word = self.vocab[int(word_int)]
            # print(word_int, word)
            char_list = list(word)
            counts = torch.zeros(self.abp_domain_size)
            if self.use_cuda:
                counts = counts.cuda()
            inputs = tuple(self.char_set[c] for c in char_list)
            targets = counts
            length = len(char_list)
            category_vector, input_vector = self.__generate_vector(
                None, inputs, None)
            rhs = (word_int,)
            for parent in pcfg_counts:
                if parent.symbol() == '0':
                    continue
                else:
                    category = int(parent.symbol()) - 1
                    count = pcfg_counts[parent].get(rhs, 0)
                    if rhs not in self.total_word_counts:
                        self.total_word_counts[rhs] = 0
                    self.total_word_counts[rhs] += count
                    counts[category] = count
            # print(counts)
            counts = torch.autograd.Variable(counts)
            this_word = RNNDiscriminativeEntry(None, word, word_int, inputs, targets, counts,
                                         length, category_vector=category_vector,
                                         input_vector=input_vector)
            entries.append(this_word)
        self.entries = RNNDiscriminativeEntryList(entries)

    def __generate_vector(self, category=None, inputs=None, outputs=None):
        c_vector, i_vector = None, None
        if category is not None:
            assert isinstance(category, int), 'category must be an integer'
            c_vector = torch.zeros(1,1)
            c_vector[0] = category
        if inputs is not None:
            assert isinstance(inputs, tuple), 'characters must a tuple'
            i_vector = torch.zeros(self.max_len, self.vocab_size)
            for index, char_id in enumerate(inputs):
                i_vector[index, char_id] = 1
        # if outputs is not None:
        #     assert isinstance(outputs, tuple), 'characters must a tuple'
        #     o_vector = torch.zeros(self.max_len, self.max_len)
        #     for index, char_id in enumerate(outputs):
        #         o_vector[index, char_id] = 1
        return c_vector, i_vector

    def _rnn_step(self):
        self.rnn.train()
        whole_input_tensor = torch.stack([entry.input_vector for entry in self.entries])
        # print(whole_input_tensor)
        whole_input_tensor = torch.autograd.Variable(whole_input_tensor)
        if self.use_cuda:
            whole_input_tensor = whole_input_tensor.contiguous().cuda()
        lengths = [entry.length for entry in self.entries]
        packed_word_tensors = torch.nn.utils.rnn.pack_padded_sequence(whole_input_tensor,
                                                                      lengths, batch_first=True)
        h_0 = self.h_0.expand(self.h_0.size(0), len(self.entries), self.h_0.size(2))
        if self.use_cuda:
            h_0 = h_0.contiguous()
        final_total_nll = 0
        first_total_nll = 0
        for iter in range(self.training_iters):
            self.optimizer.zero_grad()
            results, h_t = self.rnn.forward(packed_word_tensors, h_0)
            h_t = torch.transpose(h_t, 0, 1)
            h_t = h_t.contiguous().view(h_t.size(0), -1)
            unpacked_results = self.final_layer(h_t)
            p_p_giv_w = torch.nn.functional.softmax(unpacked_results)
            # print(unpacked_results)
            total_nll = self.entries.set_scores(p_p_giv_w)
            total_nll.backward()
            if iter == 0:
                first_total_nll = total_nll.data.cpu()[0]
            self.optimizer.step()
        final_total_nll = total_nll.data.cpu()[0]
        logging.info('The RNN final training iter has a total - loglikelihood of {} for the '
                     'corpus with {} improvement.'.format(
            total_nll.data.cpu()[0], first_total_nll - final_total_nll))

        self.rnn.eval()
        packed_word_tensors.data.detach_()
        packed_word_tensors.data.volatile = True
        results, h_t = self.rnn.forward(packed_word_tensors, h_0)
        h_t = torch.transpose(h_t, 0, 1)
        h_t = h_t.contiguous().view(h_t.size(0), -1)
        unpacked_results = self.final_layer(h_t)
        p_p_giv_w = torch.nn.functional.softmax(unpacked_results)
        total_nll = self.entries.set_scores(p_p_giv_w)
        logging.info('The RNN eval iter has a total - loglikelihood of {} for the corpus.'.format(
            total_nll.data.cpu()[0]))

    def _update_pcfg(self, pcfg_counts):
        for entry in self.entries:
            for category in range(self.abp_domain_size):
                cat = self.non_terms[category]
                child = (entry.word_int, )
                original_count = pcfg_counts[cat].get(child, 0)
                new_count = self.total_word_counts[child] * entry.prob.data[category]
                pcfg_counts[cat][child] = new_count
                logging.debug("total count is {} and this word's prob is {} for cat {}".format(
                    self.total_word_counts[child], entry.prob.data[category],
                    cat))
                logging.debug('old count is {}, new count is {} for word {} '.format(original_count,
                                                                                new_count,entry.word))