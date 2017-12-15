#!/usr/bin/env python3

import unittest
import pyximport; pyximport.install()
from HmmSampler import *
import numpy as np
import Indexer
from models import Models, Model
from uhhmm import initialize_models

class CalcRangesTest(unittest.TestCase):
    
    def test_extract_states(self):
        depth = 4
        a_max=5
        b_max=5
        g_max=15
        
        models = Models()
        models.fork = [None] * depth
        ## J models:
        models.trans = [None] * depth
        models.reduce = [None] * depth
        ## Active models:
        models.act = [None] * depth
        models.root = [None] * depth
        ## Reduce models:
        models.cont = [None] * depth
        models.exp = [None] * depth
        models.next = [None] * depth
        models.start = [None] * depth
        
        for d in range(0, depth):
            ## Doesn't really matter what params are here for our purposes:
            models.fork[d] = Model((b_max, g_max, 2), name="Fork"+str(d))
            models.trans[d] = Model((b_max, g_max, 2), name="J|F1_"+str(d))
            models.reduce[d] = Model((a_max, b_max, 2), name="J|F0_"+str(d))
            models.act[d] = Model((a_max, b_max, a_max), name="A|00_"+str(d))
            models.root[d] = Model((b_max, g_max, a_max), name="A|10_"+str(d))

            models.cont[d] = Model((b_max, g_max, b_max), name="B|11_"+str(d))
            models.exp[d] = Model((g_max, a_max, b_max), name="B|10_"+str(d))
            models.next[d] = Model((a_max, b_max, b_max), name="B|01_"+str(d))
            models.start[d] = Model((a_max, a_max, b_max), name="B|00_"+str(d))

        models.pos = Model((b_max, g_max),  name="POS")
        
        models.append(models.fork)
        models.append(models.act)
        
        indexer = Indexer.Indexer(models)
        
        f = [-1, -1, 1, -1]
        j = [-1, -1, 0, -1]
        
        a = np.array([3, 0, 0, 0])
        b = np.array([2, 0, 0, 0])
        g = 10

        cur_depth = 2
        val1 = indexer.getStateIndex(f[cur_depth], j[cur_depth], a, b, g)
        print("Val1=%d" % (val1) )
        
        g = 12
        val2 = indexer.getStateIndex(f[cur_depth], j[cur_depth], a, b, g)
        print("val2=%d" % (val2) )
        
        self.assertEqual( (val2 - val1), 2 )
        
        b[0] = 3
        val3 = indexer.getStateIndex(f[cur_depth], j[cur_depth], a, b, g)
        print("val3=%d" % (val3) )
        self.assertEqual( (val3 - val2), 5*5*5*15)
        
        f = [-1, 1, -1, -1]
        j = [-1, 0, -1, -1]
        
        ## Note: new indexer does not adapt to values of f and j at different
        ## depths -- because only one level can be 0/1 for any configuration.
        cur_depth = 1
        val4 = indexer.getStateIndex(f[cur_depth], j[cur_depth], a, b, g)
        print("val4=%d" % (val4) )
        self.assertEqual( val3, val4 )

        a = np.array([4, 1, 0, 0])
        b = np.array([1, 3, 0, 0])
        
if __name__ == "__main__":
    unittest.main()
