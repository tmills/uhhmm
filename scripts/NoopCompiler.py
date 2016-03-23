#!/usr/bin/env python3.4

from PyzmqMessage import *
import pickle

class NoopCompiler:
    def compile_and_store_models(self, models, working_dir):
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        model = PyzmqModel(models, finite=False)
        pickle.dump(model, out_file)
        out_file.close()

