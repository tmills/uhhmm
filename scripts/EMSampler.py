import torch
import pickle
from Indexer import Indexer
import numpy as np
import scipy.sparse
import tqdm
import line_profiler
import torch.nn.functional

class EMSampler:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.mat = torch.zeros(4, 100, 5).cuda()
        self.mat.fill_(0)

    def _convert_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        else:
            return tensor

    def batch_sample(self, start_index, end_index, models, sents, batch_size=40):
        self.set_models(models)
        max_len = max(map(len, sents))
        self.initialize_dynprog(batch_size, max_len)
        batch_start_index = 0
        batch_end_index = 0
        P = 0
        for i in tqdm.trange(len(sents)):
            if i - batch_start_index < batch_size and i != len(sents) - 1:
                continue
            batch_end_index = i
            sent_seg = sents[batch_start_index:batch_end_index]
            if P == 0:
                P = self.sample(sent_seg, batch_start_index)
            else:
                P += self.sample(sent_seg, batch_start_index)
            # for index, sent in enumerate(sent_seg):
            #     print(sent_seg[index])
            #     print(' '.join([x.str() for x in parses[index]]))
            batch_start_index = i

        return P

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
        self.pi : torch.cuda.sparse.FloatTensor = self.pi.coalesce()
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
        EOS_index = self.indexer.get_EOS_full()
        self.end_state = torch.zeros(self.state_size, self.batch_size)
        self.end_state = self._convert_cuda(self.end_state)
        self.end_state[EOS_index, :] = 1
        torch.cuda.synchronize()
        print(self.max_len, self.state_size, self.batch_size, type(self.max_len),type(self.state_size),
              type(self.batch_size))
        self.trellis = torch.zeros(self.max_len, self.state_size, self.batch_size)
        print(self.trellis.shape)
        self.trellis.fill_(0)
        self.trellis = self._convert_cuda(self.trellis)
        self.backward_probs = self._convert_cuda(torch.zeros(self.max_len, self.state_size, self.batch_size).fill_(0))
        self.num_abp = self.indexer.num_abp


    def multiply_lex(self, dyn_prog, word_index, sent_id, trellis, token=None, backward=False):
        if token is not None:
            token_dist = self.lex_dist[:, token]
        else:
            token_dist = 1

        if token_dist is not 1:
            if not backward:
                trellis[word_index, :, sent_id] = (dyn_prog.view(-1, 1).expand(
                self.state_size_no_g, self.indexer.num_abp) * self.full_pos_dist.view(self.state_size_no_g,
                               self.indexer.num_abp) * token_dist.unsqueeze(0).expand(self.state_size_no_g, self.indexer.num_abp)).view(-1)
            else:
                trellis[word_index, :, sent_id] = (dyn_prog.view(-1, 1).expand(
                    self.state_size_no_g, self.indexer.num_abp) * token_dist.unsqueeze(
                    0).expand(self.state_size_no_g, self.indexer.num_abp)).view(-1)
        else:
            if not backward:
                trellis[word_index, :, sent_id] = (dyn_prog.view(-1, 1).expand(
                self.state_size_no_g, self.indexer.num_abp) * self.full_pos_dist.view(self.state_size_no_g,
                               self.indexer.num_abp)).view(-1)
            else:
                trellis[word_index, :, sent_id] = (dyn_prog.view(-1, 1).expand(
                self.state_size_no_g, self.indexer.num_abp).contiguous())


    def forward(self, prev_dyn_slice, word_index, sents, trellis, backward=False):
        if not backward:
            result = self.pi @ prev_dyn_slice
        else:
            marginalized_dyn_slice_over_g = torch.sum(prev_dyn_slice.t().view(-1, self.indexer.num_abp), axis=1).view(-1, self.state_size_no_g).t()
            result = self.pi.t() @ marginalized_dyn_slice_over_g
        # result = self.pi * prev_dyn_slice_column
        for sent_id in range(self.this_batch_size):
            if len(sents[sent_id]) > word_index:
                current_token = sents[sent_id][word_index]
            else:
                current_token = None
            self.multiply_lex(result, word_index, sent_id, trellis, current_token, backward=backward)

    def em_count(self, sents): # TODO: need to also count lexicals
        M = self.pi.repeat(1, self.num_abp).view(self.state_size, self.state_size).t() @ torch.diag(self.full_pos_dist)
        P = torch.zeros(self.state_size, self.state_size)
        for sent_id, sent in enumerate(sents):
            for word_id, word in enumerate(sent):
                index_u = len(sent) - word_id - 1
                index_v_1 = word_id - 1
                token_dist = self.lex_dist[:, word_id]
                full_token_dist = token_dist.unsqueeze(0).expand(self.state_size_no_g, self.indexer.num_abp).view(-1)
                if word_id == 0:
                    v_t_1 = self.start_state[:, sent_id]
                    u_t = self.backward_probs[index_u, :, sent_id]
                else:
                    v_t_1 = self.trellis[index_v_1, :, sent_id]
                    u_t = self.backward_probs[index_u, :, sent_id]
                P_hat = torch.diag(v_t_1) @ M @ torch.diag(full_token_dist) @ torch.diag(u_t)
                P += P_hat / torch.sum(P_hat)
        return P

    def sample(self, sents, sent_index):
        assert max(map(len, sents)) <= self.max_len - 1, "max length {} should be 1 longer than the longest sent {}".format(
            self.max_len, max(map(len,sents))
        )
        self.this_batch_size = len(sents)
        back_ward_sents = [sent[::-1] for sent in sents]
        # print(type(self.trellis), type(self.backpointers))
        # print(self.trellis)
        # self.trellis.fill_(0.)
        self.backward_probs.fill_(0)
        self.trellis.fill_(0)
        for word_index in range(self.max_len):
            if word_index == 0:
                self.forward(self.start_state, word_index, sents, self.trellis)
                self.forward(self.end_state, word_index, back_ward_sents, self.backward_probs, backward=True)
            else:
                self.forward(self.trellis[word_index - 1], word_index, sents, self.trellis)
                self.forward(self.backward_probs[word_index - 1], word_index, back_ward_sents, self.backward_probs
                             , backward=True)
        # print(states, max_probs)
        counts = self.em_count(sents)
        return counts

    def collect_counts(self, P, pcfg):
        pass