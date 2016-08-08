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

class DistributedModelCompiler(FullDepthCompiler):
    
    def __init__(self, depth, work_server, gpu=False):
        FullDepthCompiler.__init__(self, depth)
        self.work_server = work_server
        ## Parent has a depth variable but it is a cython typed variable
        self.depth = depth
        self.gpu = gpu

    def compile_and_store_models(self, models, working_dir):
        indexer = Indexer(models)
        logging.info("Compiling component models into mega-HMM transition and observation matrices")

        maxes = indexer.getVariableMaxes()
        logging.info("1")
        (a_max, b_max, g_max) = maxes
        logging.info("2")
        totalK = indexer.get_state_size()
        indptr = np.zeros(totalK+1)
        logging.info("3")
        indices =  []
        data = []
        
        ## Write out raw models for workers to use to build from:
        ## First unlog them so they are in form workers expect:
        unlog_models(models, self.depth)
        fn = working_dir+"/models.bin"
        out_file = open(fn, 'wb')
        model_wrapper = ModelWrapper(ModelWrapper.COMPILE, models, self.depth)
        pickle.dump(model_wrapper, out_file)
        out_file.close()
         ## does not seem necessary
        logging.info("4")
        t0 = time.time()
        self.work_server.submitBuildModelJobs(totalK)
        logging.info("5")
        for prevIndex in range(0,totalK):
            indptr[prevIndex+1] = indptr[prevIndex]
            (local_indices, local_data) = self.work_server.get_model_row(prevIndex)
            
            indptr[prevIndex+1] += len(local_indices)
            indices.append(local_indices)
            data.append(local_data)
        logging.info("6")
        logging.info("Flattening sublists into main list")
        flat_indices = [item for sublist in indices for item in sublist]
        flat_data = [item for sublist in data for item in sublist]
            
        logging.info("Creating csr transition matrix from sparse indices")
        if self.gpu == False:
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK), dtype=np.float64)
        else:
            pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK), dtype=np.float32)
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        logging.info("Transforming and writing csc model")
        # if gpu then dumping out two models, the one used by worker should be *.bin.gpu
        pi = pi.tocsc()
        if self.gpu == True:
            lex_dist = models.lex.dist.astype(np.float32)
            model_gpu = ModelWrapper(ModelWrapper.HMM, (pi.T, lex_dist,(a_max, b_max, g_max), self.depth), self.depth)
            gpu_out_file = open(working_dir+'/models.bin.gpu', 'wb')
            logging.info("Saving GPU models for use")
            pickle.dump(model_gpu, gpu_out_file)
            gpu_out_file.close()
        relog_models(models, self.depth)
        model = ModelWrapper(ModelWrapper.HMM, (models,pi), self.depth)
        pickle.dump(model, out_file)
        out_file.close()
        nnz = pi.nnz
        pi = None

        time_spent = time.time() - t0
        logging.info("Distributed model compiler done in %d s with %d non-zero values" % (time_spent, nnz))
