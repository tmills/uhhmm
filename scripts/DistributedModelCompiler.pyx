import logging
import time
import numpy as np
import os.path
import sys
from PyzmqMessage import ModelWrapper
import scipy.sparse
from FullDepthCompiler import FullDepthCompiler, unlog_models, relog_models
import pickle
from Indexer import Indexer
import itertools
import math

PER_STATE_CONNECTION = 100

class DistributedModelCompiler(FullDepthCompiler):

    def __init__(self, depth, work_server, gpu=False, limit_depth=-1):
        FullDepthCompiler.__init__(self, depth)
        self.work_server = work_server
        ## Parent has a depth variable but it is a cython typed variable
        self.depth = depth
        self.gpu = gpu
        self.limit_depth = self.depth if limit_depth == -1 else limit_depth
        logging.basicConfig(stream=sys.stdout)

    def compile_and_store_models(self, models, working_dir, full_pi = False, viterbi=False):
        # models = pickle.load(open(working_dir+ "/ori_models.bin" ,'rb')).model[0]
        global PER_STATE_CONNECTION
        indexer = Indexer(models)
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        if viterbi:
            logging.warning("VITERBI model compilation is ON!")
        model_wrapper_type = ModelWrapper.HMM if not viterbi else ModelWrapper.VITERBI
        maxes = indexer.getVariableMaxes()
        (a_max, b_max, g_max) = maxes
        totalK = indexer.get_state_size()
        total_connection = int(PER_STATE_CONNECTION * totalK)
       # indptr = np.zeros(totalK+1)
        if self.gpu == False:
            data_type = np.float64
            data_type_bytes = 8
        else:
            data_type = np.float32
            data_type_bytes = 4
        indptr = np.zeros(totalK+1)
        indices =  np.zeros((total_connection,),dtype=data_type)
        data = np.zeros((total_connection,), dtype=data_type)
        indptr_full = np.array([])
        indices_full = np.array([])
        data_full = np.array([])
        if full_pi:
            indptr_full = np.zeros(totalK + 1)
            indices_full =  np.zeros((total_connection,),dtype=data_type)
            data_full = np.zeros((total_connection,), dtype=data_type)
        ## Write out raw models for workers to use to build from:
        ## First unlog them so they are in form workers expect:
        unlog_models(models, self.depth)
        fn = os.path.join(working_dir,'raw_models.bin')
        model_wrapper = ModelWrapper(ModelWrapper.COMPILE, models, self.depth)
        out_file = open(fn, 'wb')
        pickle.dump(model_wrapper, out_file)
        out_file.close()
         ## does not seem necessary
        t0 = time.time()
        self.work_server.submitBuildModelJobs(totalK, full_pi)
        index_data_indices = -1
        index_data_indices_full = -1
        for prevIndex in range(0,totalK):
            if prevIndex % 100000 == 0:
                logging.info("Model Compiler compiling row {} / {}".format(prevIndex, totalK))
            indptr[prevIndex+1] = indptr[prevIndex]
            if full_pi:
                indptr_full[prevIndex + 1] = indptr_full[prevIndex]
            (local_indices, local_data, local_indices_full, local_data_full) = self.work_server.get_model_row(prevIndex)
            assert len(local_indices) == len(local_data), 'Bad length match at %d' % prevIndex

            # # normalization
            # normalizer = sum(local_data)
            # if normalizer > 0:
            #     local_data = [x/normalizer for x in local_data]
            # if full_pi:
            #     normalizer = sum(local_data_full)
            #     if normalizer > 0:
            #         local_data_full = [x/normalizer for x in local_data_full]

            indptr[prevIndex+1] += len(local_indices)
            for local_indices_index, local_indices_item in enumerate(local_indices):
                index_data_indices += 1
                indices[index_data_indices] = local_indices_item
                data[index_data_indices] = local_data[local_indices_index]
            if full_pi:
                indptr_full[prevIndex + 1] += len(local_indices_full)
                for local_indices_full_index, local_indices_full_item in enumerate(local_indices_full):
                    index_data_indices_full += 1
                    indices_full[index_data_indices_full] = local_indices_full_item
                    data_full[index_data_indices_full] = local_data_full[local_indices_full_index]

        else:
            indices = indices[:index_data_indices+1]
            data = data[:index_data_indices+1]
            if full_pi:
                indices_full = indices_full[:index_data_indices_full+1]
                data_full = data_full[:index_data_indices_full+1]
            # assert data[-1] != 0. and indices[-1] != 0., '0 prob at the end of sparse Pi.'
        logging.info("Per state connection is %f" % (index_data_indices/totalK))

        # update the per state connection guess
        if PER_STATE_CONNECTION == 100:
            PER_STATE_CONNECTION = math.ceil(index_data_indices / totalK)

        logging.info("Size of PI/g will roughly be %.2f M" % ((index_data_indices * 2 + (totalK+1))*data_type_bytes / 1e6))
        # logging.info("Flattening sublists into main list")
        flat_indices = indices
        flat_data = data
        # logging.info('Last ptr %d; length of data: %d' % (indptr[-1], flat_data.shape[0]))
        logging.info("Creating csr transition matrix from sparse indices")
        if self.gpu == False:
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK / g_max), dtype=np.float64)
            if full_pi:
                pi_full = scipy.sparse.csr_matrix((data_full, indices_full, indptr_full), (totalK, totalK),
                                              dtype=np.float64)
        else:
            # logging.info("Dumping out the GPU version of PI.")
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK / g_max), dtype=np.float32)
            if full_pi:
                pi_full = scipy.sparse.csr_matrix((data_full,indices_full,indptr_full), (totalK, totalK ), dtype=np.float32)
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        # logging.info("Transforming and writing csc model")
        # if gpu then dumping out two models, the one used by worker should be *.bin.gpu
        pi = pi.tocsc()
        row_indices = list(itertools.product(range(b_max), repeat=self.depth))
        if self.gpu == True:
            lex_dist = 10**(models.lex.dist.astype(np.float32))
            pos_dist = models.pos.dist
            pos_dist[:, -1].fill(0)
            pos_dist[-1, :].fill(0)
            pos_dist[:, 0].fill(0)
            if self.depth > 1:
                corrected_pos_dist = np.zeros((pos_dist.shape[0]**self.depth, pos_dist.shape[1]))
                original_num_rows = pos_dist.shape[0]
                for index, row_index in enumerate(row_indices):
                    if any([x == b_max - 1 for x in row_index]):
                        corrected_pos_dist[index] = 0
                        continue
                    row_index_bool = [bool(x) for x in row_index]
                    boundary = -1
                    prev_val = None
                    switch = 0
                    for val_index, val in enumerate(row_index_bool):
                        if prev_val is None:
                            cur_val = val
                            prev_val = val
                        else:
                            cur_val = val
                            if (prev_val is False and cur_val is True):
                                corrected_pos_dist[index] = 0
                                break
                        prev_val = cur_val
                        if cur_val is True:
                            boundary = val_index
                    else:
                        if all([ not x for x in row_index_bool]):
                            corrected_pos_dist[index] = pos_dist[0]
                            # print(row_index, 0)
                        else:
                            corrected_pos_dist[index] = pos_dist[row_index[boundary]]
                            # print(row_index, row_index[boundary])
            else:
                corrected_pos_dist = pos_dist
            corrected_pos_dist = np.repeat(corrected_pos_dist, [2]*corrected_pos_dist.shape[0], axis=0)
            # pos_dist = np.repeat(pos_dist, 2)
            for bf_index, g_probs in enumerate(corrected_pos_dist):
                if np.sum(g_probs) != 0:
                    if bf_index % 2 == 0:
                        row_index = bf_index // 2
                        b_index = 0
                        for row_index_val in row_indices[row_index][::-1]:
                            if row_index_val > 0:
                                b_index = row_index_val
                                break
                        for g_index, g_val in enumerate(g_probs):
                            if g_index == b_index:
                                corrected_pos_dist[bf_index, g_index] = 1
                            else:
                                corrected_pos_dist[bf_index, g_index] = 0
            corrected_pos_dist = np.ravel(corrected_pos_dist.astype(np.float32))

            assert corrected_pos_dist.shape[0]== g_max*(b_max**self.depth)*2, "Size of POS array {} should be analytically" \
                                                                              " equal to {}".format(pos_dist.shape[0],
                                                                                                    g_max*(b_max**self.depth)*2)

            for b_index, b_cats in enumerate(row_indices):
                for f_index in (0, 1):
                    for g_index in range(0, g_max):
                        logging.debug(' '.join(map(str, ['P', g_index, 'B',b_cats,'F', f_index,
                                                         corrected_pos_dist[b_index*2*g_max+f_index*g_max+g_index]])))
            # for index_t_1 in pi.shape[0]:
            #     state_t_1 = indexer.extractState(index_t_1)
            #     if np.sum(pi[index_t_1]) == 0:
            #         logging.info(state_t_1.str() + ' is a invalid state.')
            #         continue
            #     for index_t in pi.shape[1]:
            #         if pi[index_t_1, index_t]:
            #             state_t = indexer.extractState(index_t)
            #             logging.info(' '.join(map(str, [state_t_1.str(), '->', state_t.str(), pi[index_t_1, index_t],
            #                                             'pos line:', pos_dist[]])))
            model_gpu = ModelWrapper(model_wrapper_type,
                                         (pi.T, lex_dist, (a_max, b_max, g_max), self.depth, corrected_pos_dist,
                                          indexer.get_EOS_full()), self.depth)
            # logging.info("EOS index is "+str(indexer.get_EOS_full()))
            gpu_model_file_name = working_dir+'/models.bin.gpu'
            if viterbi:
                gpu_model_file_name += '.viterbi'
            gpu_out_file = open(gpu_model_file_name, 'wb')
            # logging.info("Saving GPU models for use")
            pickle.dump(model_gpu, gpu_out_file)
            gpu_out_file.close()
            if viterbi:
                return gpu_model_file_name
        relog_models(models, self.depth)

        if full_pi:
            pi_full = pi_full.tocsc()
            model = ModelWrapper(model_wrapper_type, (models, pi_full, models.ac_coeff), self.depth)
        else:
            model = ModelWrapper(model_wrapper_type, (models, pi), self.depth)
        # EOS = indexer.get_EOS()
        # EOS_full = indexer.get_EOS_full()
        # EOS_1wrd = indexer.get_EOS_1wrd()
        # EOS_1wrd_full = indexer.get_EOS_1wrd_full()
#        print(pi[EOS_full,:].sum())
#         print(pi.shape)
#         print(pi[:,EOS].sum())
#        print(pi[EOS_1wrd_full,:].sum())
#        print(pi[:,EOS_1wrd].sum())
        pickle.dump(model, out_file)
        out_file.close()
        nnz = pi.nnz
        pi = None
        pi_full = None
        time_spent = time.time() - t0
        logging.info("Distributed model compiler done in %d s with %d non-zero values" % (time_spent, nnz))
