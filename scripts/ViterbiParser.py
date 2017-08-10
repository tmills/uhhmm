import torch
import pickle
from Indexer import Indexer
import numpy as np
import scipy.sparse
import tqdm
import line_profiler

class ViterbiParser:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.mat = torch.zeros(4, 100, 5).cuda()
        self.mat.fill_(0)

    def _convert_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        else:
            return tensor

    def batch_parse(self, start_index, end_index, models, sents, batch_size=40):
        self.set_models(models)
        max_len = max(map(len, sents))
        self.initialize_dynprog(batch_size, max_len)
        batch_start_index = 0
        batch_end_index = 0
        results = []
        for i in tqdm.trange(len(sents)):
            if i - batch_start_index < batch_size and i != len(sents) - 1:
                continue
            batch_end_index = i
            sent_seg = sents[batch_start_index:batch_end_index]
            parses, _ = self.parse(sent_seg, batch_start_index)
            assert all([len(x[0]) == len(x[1]) for x in zip(sent_seg, parses)])
            # for index, sent in enumerate(sent_seg):
            #     print(sent_seg[index])
            #     print(' '.join([x.str() for x in parses[index]]))
            results.extend(parses)
            batch_start_index = i

        return results

    def set_models(self, models):
        self.mat = torch.zeros(4, 100, 5).cuda()
        self.mat.fill_(0)
        self.pi, self.lex_dist, self.maxes, self.depth, self.pos_dist, self.EOS_index = models.model
        self.pi = self.pi.tocoo()
        pi_shape = self.pi.shape
        indices = torch.from_numpy(np.vstack([self.pi.row,self.pi.col])).long()

        values = torch.from_numpy(self.pi.data)
        # print(indices.shape, values.shape, self.pi.shape)
        self.pi = torch.sparse.FloatTensor(indices, values, torch.Size(pi_shape))
        self.mat = torch.zeros(4, 100, 5).cuda()
        self.mat.fill_(0)
        self.mat = torch.zeros(4, 100, 5).cuda()
        self.mat.fill_(0)
        self.indexer = Indexer({'depth':self.depth, 'num_abp':self.maxes[0]})
        self.state_size = self.indexer.get_state_size()
        self.state_size_no_g = int(self.indexer.get_state_size() / self.indexer.num_abp)
        self.full_pos_dist = self.make_full_pos_array(self.pos_dist)

        self.pi = self._convert_cuda(self.pi)
        self.pi = self.pi.coalesce()
        self.lex_dist = self._convert_cuda(torch.from_numpy(self.lex_dist))
        self.full_pos_dist = self._convert_cuda(self.full_pos_dist)

    def make_full_pos_array(self, pos_array):
        if isinstance(pos_array, np.ndarray):
            pos_array = torch.from_numpy(pos_array)
        # print(pos_array.shape)
        # print(pos_array[:10])
        pos_array = torch.squeeze(pos_array)
        pos_array_size = pos_array.numel()
        num_rows = int(self.state_size / pos_array_size)
        # print(num_rows)
        pos_array = torch.unsqueeze(pos_array, 0)
        pos_array = (pos_array.expand(num_rows, pos_array_size).contiguous()).view(-1)
        # print(pos_array.shape)
        # print(pos_array[:10])
        return pos_array

    def initialize_dynprog(self, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = int(max_len + 1)
        self.start_state = torch.zeros(self.state_size, self.batch_size)
        self.start_state = self._convert_cuda(self.start_state)
        self.start_state[0, :] = 1
        torch.cuda.synchronize()
        print(self.max_len, self.state_size, self.batch_size, type(self.max_len),type(self.state_size),
              type(self.batch_size))
        self.trellis = torch.zeros(self.max_len, self.state_size, self.batch_size).cuda()
        print(self.trellis.shape)
        self.trellis.fill_(0)
        # self.trellis = self._convert_cuda(self.trellis)
        self.trellis = self.trellis.cuda()
        self.trellis.fill_(0)
        self.backpointers = self._convert_cuda(torch.zeros(self.max_len, self.state_size, self.batch_size).long().fill_(-1))
        self.shrunk_trellis_column = self._convert_cuda(torch.zeros(self.state_size_no_g))
        self.shrunk_backpointers_column = self._convert_cuda(torch.zeros(self.state_size_no_g))
        # print(self.trellis.shape, self.backpointers.shape, self.shrunk_backpointers_column.shape, self.shrunk_trellis_column.shape)
        # print(type(self.trellis), type(self.backpointers))

    # not efficient. must be optimized
    @profile
    def viterbi_max(self, sparse_m, word_index, sent_id, token=None):
        if token is not None:
            # print(self.lex_dist.shape)
            token_dist = self.lex_dist[:, token]
            # token_dist = self.make_full_pos_array(token_dist)
        else:
            token_dist = 1
        sparse_m_rows = sparse_m.size()[0]
        assert self.state_size_no_g == sparse_m_rows, (self.state_size_no_g, sparse_m_rows, self.state_size)
        for i in range(sparse_m_rows):
            this_row = sparse_m._values()[sparse_m._indices()[0, :] == i]
            if len(this_row.shape) > 0:
                max_val, max_sparse_index = torch.max(this_row, 0)
                max_sparse_index = sparse_m._indices()[1, :][sparse_m._indices()[0, :] == i][max_sparse_index]
                self.shrunk_trellis_column[i] = max_val[0]
                self.shrunk_backpointers_column[i] = max_sparse_index[0]
        # self.trellis[word_index, :, sent_id] = self.shrunk_trellis_column.view(-1, 1).expand(
        #     self.state_size_no_g, self.indexer.num_abp).contiguous().view(-1) * self.full_pos_dist * token_dist
        # self.backpointers[word_index, :, sent_id] = self.shrunk_backpointers_column.view(-1, 1).expand(
        #     self.state_size_no_g, self.indexer.num_abp).contiguous().view(-1)
        if token_dist is not 1:
            self.trellis[word_index, :, sent_id] = (self.shrunk_trellis_column.view(-1, 1).expand(
            self.state_size_no_g, self.indexer.num_abp) * self.full_pos_dist.view(self.state_size_no_g,
                           self.indexer.num_abp) * token_dist.unsqueeze(0).expand(self.state_size_no_g, self.indexer.num_abp)).view(-1)
        else:
            self.trellis[word_index, :, sent_id] = (self.shrunk_trellis_column.view(-1, 1).expand(
            self.state_size_no_g, self.indexer.num_abp) * self.full_pos_dist.view(self.state_size_no_g,
                           self.indexer.num_abp)).view(-1)
        self.backpointers[word_index, :, sent_id] = self.shrunk_backpointers_column.view(-1, 1).expand(
            self.state_size_no_g, self.indexer.num_abp).contiguous().view(-1)
        self.shrunk_trellis_column.zero_()
        self.shrunk_backpointers_column.fill_(-1)

    def viterbi_forward(self, prev_dyn_slice, word_index, sents):
        for sent_id in range(self.this_batch_size):
            if len(sents[sent_id]) > word_index:
                current_token = sents[sent_id][word_index]
            else:
                current_token = None
            indices = self.pi._indices()[1, :]
            assert self.pi.shape[1] == prev_dyn_slice[:,sent_id].shape[0], (self.pi.shape[1], prev_dyn_slice[:,sent_id].shape[0])
            prev_dyn_slice_column = torch.index_select(prev_dyn_slice[:,sent_id], 0, indices).contiguous()
            prev_dyn_slice_sparse = torch.cuda.sparse.FloatTensor(self.pi._indices(), prev_dyn_slice_column, self.pi.shape)
            # prev_dyn_slice_sparse = prev_dyn_slice_column._sparse_mask(self.pi)
            result = self.pi * prev_dyn_slice_sparse
            # result = self.pi * prev_dyn_slice_column
            self.viterbi_max(result, word_index, sent_id, current_token)


    def viterbi_backward(self, sents):
        states = [[] for i in range(self.this_batch_size)]
        max_probs = [0 for i in range(self.this_batch_size)]
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
        assert max(map(len, sents)) <= self.max_len - 1, "max length {} should be 1 longer than the longest sent {}".format(
            self.max_len, max(map(len,sents))
        )
        self.this_batch_size = len(sents)
        # print(type(self.trellis), type(self.backpointers))
        # print(self.trellis)
        # self.trellis.fill_(0.)
        self.backpointers.fill_(0)
        for word_index in range(self.max_len):
            if word_index == 0:
                self.viterbi_forward(self.start_state, word_index, sents)
            else:
                self.viterbi_forward(self.trellis[word_index - 1], word_index, sents)
        states, max_probs = self.viterbi_backward(sents)
        # print(states, max_probs)
        return states, max_probs

