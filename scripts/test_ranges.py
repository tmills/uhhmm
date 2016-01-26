#!/usr/bin/env python3

import unittest
import pyximport; pyximport.install()
from HmmSampler import *
import numpy as np

class CalcRangesTest(unittest.TestCase):
    
    def test_extract_states(self):
        f = [-1, -1, 1, -1]
        j = [-1, -1, 0, -1]
        
        a = [3, 0, 0, 0]
        b = [2, 0, 0, 0]
        g = 10

        val1 = getStateIndex(f, j, a, b, g, (5, 5, 15))
        
        g = 12
        val2 = getStateIndex(f, j, a, b, g, (5, 5, 15))
        
        self.assertEqual( (val2 - val1), 2 )
        
        b[0] = 3
        val3 = getStateIndex(f, j, a, b, g, (5, 5, 15))
        self.assertEqual( (val3 - val2), 5*5*5*15)
        
        f = [-1, 1, -1, -1]
        j = [-1, 0, -1, -1]
        
        val4 = getStateIndex(f, j, a, b, g, (5, 5, 15))
        self.assertNotEqual( val3, val4 )

if __name__ == "__main__":
    unittest.main()
