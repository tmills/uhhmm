import torch
import pickle
from Indexer import Indexer
import numpy as np
import scipy.sparse

class ViterbiParser:
    def __init__(self, cuda=True):
        self.cuda = cuda

    def _convert_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        else:
            return tensor

    def set_models(self, models):
        with open(models,'rb') as m:
            self.pi, self.lex_dist, self.maxes, self.depth, self.pos_dist, self.EOS_index = pickle.load(m)
            self.pi = self.pi.tocoo()
            pi_shape = self.pi.shape
            indices = torch.from_numpy(np.vstack([self.pi.row, self.pi.col])).long()
            values = torch.from_numpy(self.pi.data)
            self.pi = torch.sparse.FloatTensor(indices, values, pi_shape)
        self.indexer = Indexer({'depth':self.depth, 'numabp':self.maxes[0]})
        self.state_size = self.indexer.get_state_size()
        self.state_size_no_g = self.indexer.get_state_size() / self.indexer.g_max
        self.full_pos_dist = self.make_full_pos_array(self.pos_dist)

        self.pi = self._convert_cuda(self.pi)
        self.lex_dist = self._convert_cuda(self.lex_dist)
        self.full_pos_dist = self._convert_cuda(self.full_pos_dist)

    def make_full_pos_array(self, pos_array):
        pos_array = np.ravel(pos_array)
        pos_array_size = np.cumprod(pos_array.shape)
        num_rows = self.state_size / pos_array_size
        pos_array = torch.from_numpy(np.ravel(np.repeat(pos_array, (num_rows, 1))))
        return pos_array

    def initialize_dynprog(self, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        self.start_state = torch.zeros((self.state_size, self.batch_size))
        self.start_state[0, :] = 1
        self.trellis = self._convert_cuda(torch.zeros((self.max_len, self.state_size, self.batch_size)))
        self.backpointers = self._convert_cuda(torch.zeros((self.max_len, self.state_size, self.batch_size)).fill_(-1))
        self.shrunk_trellis_column = self._convert_cuda(torch.zeros((self.state_size_no_g,)))
        self.shrunk_backpointers_column = self._convert_cuda(torch.zeros((self.state_size_no_g,)))

    def viterbi_max(self, sparse_m, word_index, sent_id, token=None):
        if token is not None:
            token_dist = self.lex_dist[:, token]
        else:
            token_dist = 1
        sparse_m_rows = sparse_m.size()[0]
        assert self.state_size_no_g == sparse_m_rows, (self.state_size_no_g, sparse_m_rows, self.state_size)
        for i in range(sparse_m_rows):
            this_row = sparse_m.values()[sparse_m.indices()[:, 0] == i]
            if this_row:
                max_val, max_sparse_index = torch.max(this_row, 0)
                max_sparse_index = sparse_m.indices[:, 1][sparse_m.indices()[:, 0] == i][max_sparse_index]
                self.shrunk_trellis_column[i] = max_val
                self.shrunk_backpointers_column[i] = max_sparse_index
        self.trellis[word_index, :, sent_id] = self.shrunk_trellis_column.view(-1, 1).expand(
            self.state_size_no_g, self.indexer.g_max).view(-1, 1).contiguous() * self.full_pos_dist * token_dist
        self.backpointers[word_index, :, sent_id] = self.shrunk_backpointers_column.view(-1, 1).expand(
            self.state_size_no_g, self.indexer.g_max).view(-1, 1).contiguous()
        self.shrunk_trellis_column.zero_()
        self.shrunk_backpointers_column.fill_(-1)

    def viterbi_forward(self, prev_dyn_slice, word_index, sents):
        for sent_id in range(self.batch_size):
            if len(sents[sent_id]) >= word_index:
                current_token = sents[sent_id][word_index]
            else:
                current_token = None
            prev_dyn_slice_column = prev_dyn_slice[:,sent_id].expand_as(self.pi)
            prev_dyn_slice_sparse = prev_dyn_slice_column.sparse_mask(self.pi)
            result = self.pi * prev_dyn_slice_sparse
            self.viterbi_max(result, word_index, sent_id, current_token)


    def viterbi_backward(self, sents):
        states = [[] for i in range(self.batch_size)]
        max_probs = [0 for i in range(self.batch_size)]
        EOS_index = self.indexer.get_EOS_full()
        for sent_id, sent in enumerate(sents):
            sent_len = len(sent)
            for i in range(len(sents[sent_id]),0,-1):
                if i == len(sents[sent_id]):
                    states[sent_id].append(self.backpointers[i, EOS_index, sent_id])
                    max_probs[sent_id] = np.log10(self.trellis[i, EOS_index, sent_id])
                else:
                    states[sent_id].append(self.backpointers[i, states[sent_id][-1], sent_id])
        states = [x[::-1] for x in states]
        states = [[ self.indexer.extractState(id) for id in x] for x in states]
        return states, max_probs

    def parse(self, sents, sent_index):
        assert max(map(len, sents)) == self.max_len + 1, "max length should be 1 longer than the longest sent"
        self.trellis.zero_()
        self.backpointers.zero_()
        for word_index in range(self.max_len):
            if word_index == 0:
                self.viterbi_forward(self.start_state, word_index, sents)
            else:
                self.viterbi_forward(self.trellis[word_index - 1], word_index, sents)
        states, max_probs = self.viterbi_backward(sents)
        return states, max_probs

    def sample(self, sents, sent_index, posterior_decoding): # conforming to the same API, not really useful
        return self.parse(sents, sent_index)

