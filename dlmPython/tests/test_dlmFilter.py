from unittest import TestCase

from dlmPython import dlm_uni_df, lego, param
import numpy as np
from scipy.linalg import block_diag


class TestDLM(TestCase):

    ### tests for concatenating delta and dim ###
    def test_join(self):
        np.testing.assert_equal(lego.join(1,1), np.array([1,1]))
        np.testing.assert_equal(lego.join(1.,1), np.array([1,1]))
        np.testing.assert_equal(lego.join(1.,1.), np.array([1,1]))

        np.testing.assert_equal(
            lego.join(lego.E2(3), lego.E2(3)),
                      np.array([1, 0, 0, 1, 0, 0]))

        np.testing.assert_equal(
            lego.join(lego.E2(3), 1),
                      np.array([1, 0, 0, 1]))

        np.testing.assert_equal(
            lego.join(1, lego.E2(3)),
                      np.array([1, 1, 0, 0]))


    def test_dlm_add(self):
        dlm1 = dlm_uni_df(lego.E2(3), lego.J(3), V=1, delta=1)
        dlm2 = dlm_uni_df(lego.E2(3), lego.J(3), V=1, delta=1)
        dlm3 = dlm1 + dlm2

        # Test F
        np.testing.assert_equal( 
                dlm3.F,
                np.array([1,0,0,1,0,0]))

        # Test G
        np.testing.assert_equal( 
                dlm3.G, 
                block_diag(lego.J(3), lego.J(3)))

        # Test V
        self.assertTrue(dlm3.V == dlm1.V + dlm2.V)

        # Test dim
        np.testing.assert_equal( 
                dlm3.dim, 
                np.array([3,3]))

        # Test delta
        np.testing.assert_equal( 
                dlm3.delta, 
                np.array([1,1]))

        #print dlm3

    # LINEAR TREND
    def test_dlm_uni_df_filter(self):
        c = dlm_uni_df(
                F=lego.E2(2), G=lego.J(2), V=1, 
                delta=.95)
        n = 30
        nAhead = 5
        y = np.array(range(n)) + np.random.normal(0, .1, n)
        init = param.uni_df(
                m=np.asmatrix(np.zeros((2,1))), 
                C=np.eye(2))
        out = c.filter(y, init)
        print out[-1].f


