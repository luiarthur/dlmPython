from unittest import TestCase

from dlmPython import dlm_uni, lego, param, dlm_mod
import numpy as np
from scipy.linalg import block_diag


class Test_dlm_uni(TestCase):

    ### tests for concatenating delta and dim ###
    def test_join(self):
        np.testing.assert_equal(lego.join(1,1), np.array([1,1]))
        np.testing.assert_equal(lego.join(1.,1), np.array([1,1]))
        np.testing.assert_equal(lego.join(1.,1.), np.array([1,1]))

        np.testing.assert_equal(
            lego.join(lego.E(3), lego.E(3)),
                      np.array([1, 0, 0, 1, 0, 0]))

        np.testing.assert_equal(
            lego.join(lego.E(3), 1),
                      np.array([1, 0, 0, 1]))

        np.testing.assert_equal(
            lego.join(1, lego.E(3)),
                      np.array([1, 1, 0, 0]))


    def test_mod_poly(self):
        dlm1 = dlm_mod.poly(1,V=3)
        dlm2 = dlm_mod.poly(2, V=3, discount=1)
        dlm3 = dlm_uni(lego.E(4), lego.J(4)*2, V=2)
        dlm = dlm1 + dlm2 + dlm3

        # Test get_block
        np.testing.assert_equal(
                dlm.__get_block__(dlm.G, 0),
                lego.J(2))
        np.testing.assert_equal(
                dlm.__get_block__(dlm.G, 2),
                2*lego.J(4))


    def test_dlm_add(self):
        dlm1 = dlm_mod.poly(2, discount=1)
        dlm2 = dlm_mod.arma(ar=[3,2,4], ma=[1,2,3], tau2=3)
        dlm3 = dlm_uni(lego.E(2), lego.J(2))
        dlm = dlm1 + dlm2 + dlm3
        print dlm

        ## Test F
        #np.testing.assert_equal( 
        #        dlm.F,
        #        np.array([1,0,0,1,0,0,0,1,0]))

        ## Test G
        #np.testing.assert_equal( 
        #        dlm3.G, 
        #        block_diag(lego.J(3), lego.J(4)))

        ## Test dim
        #np.testing.assert_equal( 
        #        dlm3.dim, 
        #        np.array([3,3]))

        ## Test delta
        #np.testing.assert_equal( 
        #        dlm3.delta, 
        #        np.array([1,1]))

        ##print dlm3


