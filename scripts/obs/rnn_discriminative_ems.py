import logging

import torch
from typing import List
import torch.optim
from .rnn_generative_ems import RNNEntry

class RNNDiscriminativeEntry(RNNEntry):
    def get_nll(self):
        return 0 - self.count * torch.log(self.prob)

class RNNEntryList:
    def __init__(self, entries : List[RNNEntry]):
        self.entries = entries
        self.entries.sort(key=lambda x: x.length, reverse=True)

    def __iter__(self):
        for entry in self.entries:
            yield entry

    def __len__(self):
        return len(self.entries)

    def set_scores(self, scores, w_logistic, b_logistic):
        for entry_index, entry in enumerate(self.entries):
            score = 0
            for target_index, target in enumerate(entry.target):
                score += scores[entry_index, target_index, target] * w_logistic[target_index] + \
                         b_logistic[target_index]
            entry.set_exp_score(score)
        category_list = []
        category_scores = []
        for entry_index, entry in enumerate(self.entries):
            if entry.category not in category_list:
                category_list.append(entry.category)
                category_scores.append(0)
            category_scores[category_list.index(entry.category)] += entry.score
        total_nll = 0
        for entry_index, entry in enumerate(self.entries):
            this_total_score = category_scores[category_list.index(entry.category)]
            prob = entry.score / this_total_score
            entry.set_prob(prob)
            total_nll += entry.get_nll()
        return total_nll

class RNNGenerativeEmission(torch.nn.Module):
    def __init__(self, abp_domain_size, vocab, hidden_size=50, num_layers=1, use_cuda=True):
        super(RNNGenerativeEmission, self).__init__()
        self.use_cuda = use_cuda
        self.vocab = vocab
        # print(vocab)
        self.vocab_size = 0
        self.char_set = {}
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = 0
        self._generate_vocab()
        self.abp_domain_size = abp_domain_size + 1
        self.input_size = len(self.char_set) + abp_domain_size +1
        self.total_rule_counts = {}
        self.non_terms = {}
        self.rnn = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)
        self.final_layer = torch.nn.Linear(self.hidden_size, self.vocab_size)
        self.h_0 = torch.nn.Parameter(data=torch.zeros(
            self.num_layers, 1, hidden_size))
        self.w_logistic = torch.nn.Parameter(data=torch.ones(self.max_len))
        self.b_logistic = torch.nn.Parameter(data=torch.zeros(self.max_len))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        print(self)

    def _generate_vocab(self):
        chars = {}
        chars['<bow>'] = len(chars)
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
        self.total_rule_counts = {}
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
            if parent.symbol() == '0':
                continue
            else:
                category = int(parent.symbol())
                for child in pcfg_counts[parent]:
                    if len(child) == 1:  #unary branching
                        count = pcfg_counts[parent][child]
                        if parent not in self.total_rule_counts:
                            self.total_rule_counts[parent] = 0
                            self.non_terms[int(parent.symbol())] = parent
                        self.total_rule_counts[parent] += count
                        word_int = child[0]
                        word = self.vocab[int(word_int)]
                        char_list = ['<bow>',] + list(word)
                        inputs = tuple(self.char_set[c] for c in char_list[:-1])
                        targets = tuple(self.char_set[c] for c in char_list[1:])
                        length = len(list(word))
                        category_vector, input_vector = self.__generate_vector(
                            category, inputs, targets)
                        this_word = RNNEntry(category, word, word_int, inputs, targets, count,
                                             length, category_vector=category_vector,
                                             input_vector=input_vector)
                        entries.append(this_word)

        self.entries = RNNEntryList(entries)

    def __generate_vector(self, category=None, inputs=None, outputs=None):
        c_vector, i_vector = None, None
        if category is not None:
            assert isinstance(category, int), 'category must be an integer'
            c_vector = torch.zeros(self.max_len, self.abp_domain_size)
            c_vector[:, category] = 1
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
        self.optimizer.zero_grad()
        whole_input_tensor = torch.stack([torch.cat([entry.category_vector, entry.input_vector], 1)
                                          for entry in self.entries])
        # print(whole_input_tensor)
        whole_input_tensor = torch.autograd.Variable(whole_input_tensor)
        if self.use_cuda:
            whole_input_tensor = whole_input_tensor.contiguous().cuda()
        lengths = [entry.length for entry in self.entries]
        assert len(lengths) == len(self.entries)
        packed_word_tensors = torch.nn.utils.rnn.pack_padded_sequence(whole_input_tensor,
                                                                      lengths, batch_first=True)
        h_0 = self.h_0.expand(self.h_0.size(0), len(self.entries), self.h_0.size(2))
        if self.use_cuda:
            h_0 = h_0.contiguous()
        results, h_t = self.rnn.forward(packed_word_tensors, h_0)
        unpacked_results, _ = torch.nn.utils.rnn.pad_packed_sequence(results, batch_first=True)
        unpacked_results = self.final_layer(unpacked_results)
        # print(unpacked_results)
        total_nll = self.entries.set_scores(unpacked_results, self.w_logistic, self.b_logistic)
        print(total_nll)
        total_nll.backward()
        self.optimizer.step()

        packed_word_tensors.data.detach_()
        packed_word_tensors.data.volatile = True
        results, h_t = self.rnn.forward(packed_word_tensors, h_0)
        unpacked_results, _ = torch.nn.utils.rnn.pad_packed_sequence(results, batch_first=True)
        unpacked_results = self.final_layer(unpacked_results)
        total_nll = self.entries.set_scores(unpacked_results, self.w_logistic, self.b_logistic)
        logging.info('The RNN has a total - loglikelihood of {} for the corpus.'.format(total_nll[
                                                                                           0]))
        print(self.w_logistic)
        print(self.b_logistic)

    def _update_pcfg(self, pcfg_counts):
        for entry in self.entries:
            cat = self.non_terms[entry.category]
            child = (entry.word_int,)
            original_count = pcfg_counts[cat][child]
            new_count = self.total_rule_counts[cat] * entry.prob.data[0]
            pcfg_counts[cat][child] = new_count
            logging.debug("total count is {} and this word's prob is {} for cat {}".format(
                self.total_rule_counts[cat], entry.prob.data[0],
                entry.category))
            logging.debug('old count is {}, new count is {} for word {} '.format(original_count,
                                                                            new_count,entry.word))