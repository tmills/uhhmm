#!/usr/bin/env python3.4

import unittest
import calcV
import numpy as np

class CalcVTest(unittest.TestCase):
#    def setUp(self):
        ## Nothing yet
    
#    def tearDown(self):
        ## Nothing yet
    
    def test_get_distribution(self):
        self.assertTrue(True)
        l1 = [[0,1,2,3]]
        d1 = calcV.get_distribution(l1)
        d1_expected = np.array([1./4, 1./4, 1./4, 1./4])
        self.assertAlmostEqual(d1.sum(), 1.0)
        #self.assertAlmostEqual(d1[0], d1_expected[0])
        self.assertListAlmostEqual(d1, d1_expected)
        
        l2 = [[0,1,2,3],[0,4,2,2]]
        d2 = calcV.get_distribution(l2)
        self.assertAlmostEqual(d2.sum(), 1.0)
        d2_expected = np.array([0.25, 0.125, 0.375, 0.125, 0.125])
        self.assertListAlmostEqual(d2, d2_expected)
        
        l3 = [[3,4,3],[3,3,6,7,8]]
        d3 = calcV.get_distribution(l3)
        self.assertAlmostEqual(d3.sum(), 1.0)
        d3_expected = np.array([0, 0, 0, 0.5, 0.125, 0, 0.125, 0.125, 0.125])
        self.assertListAlmostEqual(d3, d3_expected)
            
    def test_get_joint_distribution(self):
        self.assertTrue(True)
        l1 = [[0,1,2,3],[4,5,6,7],[0,1,7,7]]
        l2 = [[0,2,1,1],[4,4,3,8],[0,7,6,6]]
        xy = calcV.get_joint_distribution(l1, l2)
        self.assertAlmostEqual(xy.sum(), 1.0)
        self.assertAlmostEqual(xy[0][0], 2./12)
        self.assertAlmostEqual(xy[0][1], 0)
        self.assertAlmostEqual(xy[7][6], 2./12)
        self.assertAlmostEqual(xy[6][7], 0)
        self.assertAlmostEqual(xy[4][4], 1./12)
        
    def assertListAlmostEqual(self, l1, l2):
        self.assertTrue(len(l1) == len(l2))
        for ind,val in enumerate(l1):
            self.assertAlmostEqual(l1[ind], l2[ind])
        
        
if __name__ == "__main__":
    unittest.main()
