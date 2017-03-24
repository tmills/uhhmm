#!/usr/bin/env python3

import logging
import time
import numpy as np
import sys
from PyzmqMessage import ModelWrapper
import scipy.sparse
from FullDepthCompiler import FullDepthCompiler, unlog_models, relog_models
import pickle
from Indexer import Indexer
import itertools

class DistributedModelCompiler(FullDepthCompiler):
    
    def __init__(self, depth, work_server, gpu=False, limit_depth=-1):
        FullDepthCompiler.__init__(self, depth)
        self.work_server = work_server
        ## Parent has a depth variable but it is a cython typed variable
        self.depth = depth
        self.gpu = gpu
        self.limit_depth = self.depth if limit_depth == -1 else limit_depth

    def compile_and_store_models(self, models, working_dir, per_state_connection_guess = 600, full_pi = False):
        if self.gpu == False:
            full_pi = True
        # models = pickle.load(open(working_dir+ "/ori_models.bin" ,'rb')).model[0]
        indexer = Indexer(models)
        logging.info("Compiling component models into mega-HMM transition and observation matrices")

        maxes = indexer.getVariableMaxes()
        (a_max, b_max, g_max) = maxes
        totalK = indexer.get_state_size()
        total_connection = per_state_connection_guess * totalK
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
        fn = working_dir+"/models.bin"
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
            if prevIndex % 10000 == 0:
                logging.info("Model Compiler compiling row %d" % prevIndex)
            indptr[prevIndex+1] = indptr[prevIndex]
            if full_pi:
                indptr_full[prevIndex + 1] = indptr_full[prevIndex]
            (local_indices, local_data, local_indices_full, local_data_full) = self.work_server.get_model_row(prevIndex)
            assert len(local_indices) == len(local_data), 'Bad length match at %d' % prevIndex
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
        logging.info("Per state connection is %d" % (index_data_indices/totalK))
        logging.info("Size of PI/g will roughly be %.2f M" % ((index_data_indices * 2 + (totalK+1))*data_type_bytes / 1e6))
        logging.info("Flattening sublists into main list")
        flat_indices = indices
        flat_data = data
        logging.info('Last ptr %d; length of data: %d' % (indptr[-1], flat_data.shape[0]))
        logging.info("Creating csr transition matrix from sparse indices")
        if self.gpu == False:
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK / g_max), dtype=np.float64)
            if full_pi:
                pi_full = scipy.sparse.csr_matrix((data_full, indices_full, indptr_full), (totalK, totalK),
                                              dtype=np.float64)
        else:
            logging.info("Dumping out the GPU version of PI.")
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK / g_max), dtype=np.float32)
            if full_pi:
                pi_full = scipy.sparse.csr_matrix((data_full,indices_full,indptr_full), (totalK, totalK ), dtype=np.float32)
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        logging.info("Transforming and writing csc model")
        # if gpu then dumping out two models, the one used by worker should be *.bin.gpu
        pi = pi.tocsc()
        row_indices = list(itertools.product(range(b_max), repeat=self.depth))
        if self.gpu == True:
            lex_dist = 10**(models.lex.dist.astype(np.float32))
            pos_dist = models.pos.dist
            pos_dist[:, -1].fill(0)
            pos_dist[-1, :].fill(0)
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
                        if all(row_index_bool) is False:
                            corrected_pos_dist[index] = pos_dist[0]
                        else:
                            corrected_pos_dist[index] = pos_dist[row_index[boundary]]
            else:
                corrected_pos_dist = pos_dist
            pos_dist = np.ravel(corrected_pos_dist.astype(np.float32))
            pos_dist = np.repeat(pos_dist, 2)
            full_g_num = g_max * 2
            for prob_index, prob in enumerate(pos_dist):
                if prob != 0:
                    if prob_index % 2 == 0:
                        row_index = prob_index // full_g_num
                        p_index = (prob_index % (full_g_num)) // 2
                        b_val = 0
                        for row_index_val in row_indices[row_index][::-1]:
                            if row_index_val > 0:
                                b_val = row_index_val
                                break
                        if b_val != p_index:
                            pos_dist[prob_index] = 0
                        else:
                            pos_dist[prob_index] = 1
            for index_, val in enumerate(pos_dist):
                print index_, row_indices[index_//2],val
                logging.debug(' '.join(map(str, [index_, row_indices[index_//2],val])))
            model_gpu = ModelWrapper(ModelWrapper.HMM, (pi.T, lex_dist,(a_max, b_max, g_max), self.depth, pos_dist,
                                                        indexer.get_EOS_full()), self.depth)
            logging.info("EOS index is "+str(indexer.get_EOS_full()))
            gpu_out_file = open(working_dir+'/models.bin.gpu', 'wb')
            logging.info("Saving GPU models for use")
            pickle.dump(model_gpu, gpu_out_file)
            gpu_out_file.close()
        relog_models(models, self.depth)
        if full_pi:
            pi_full = pi_full.tocsc()
            model = ModelWrapper(ModelWrapper.HMM, (models, pi_full), self.depth)
        else:
            model = ModelWrapper(ModelWrapper.HMM, (models, pi), self.depth)
        # EOS = indexer.get_EOS()
        # EOS_full = indexer.get_EOS_full()
        # EOS_1wrd = indexer.get_EOS_1wrd()
        # EOS_1wrd_full = indexer.get_EOS_1wrd_full()
#        print(pi[EOS_full,:].sum())
#        print(pi[:,EOS].sum())
#        print(pi[EOS_1wrd_full,:].sum())
#        print(pi[:,EOS_1wrd].sum())
        pickle.dump(model, out_file)
        out_file.close()
        nnz = pi.nnz
        pi = None
        pi_full = None
        time_spent = time.time() - t0
        logging.info("Distributed model compiler done in %d s with %d non-zero values" % (time_spent, nnz))
