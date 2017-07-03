from unittest import TestCase

from dlmPython import dlm_uni_df, dlm_uni, lego, param, dlm_mod
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

class TestDLM(TestCase):

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


    def test_dlm_add(self):
        dlm1 = dlm_uni_df(lego.E(3), lego.J(3), delta=1)
        dlm2 = dlm_uni_df(lego.E(3), lego.J(3), delta=1)
        dlm3 = dlm1 + dlm2

        # Test F
        np.testing.assert_equal( 
                dlm3.F,
                np.array([1,0,0,1,0,0]))

        # Test G
        np.testing.assert_equal( 
                dlm3.G, 
                block_diag(lego.J(3), lego.J(3)))

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
                F=lego.E(2), G=lego.J(2), delta=.95)
        n = 30
        nAhead = 5
        y = np.array(range(n)) + np.random.normal(0, .1, n)
        init = param.uni_df(
                m=np.asmatrix(np.zeros((2,1))), 
                C=np.eye(2))
        filt = c.filter(y, init)
        print
        print y[-1]
        print filt[-1].f

    def test_dlm_uni_df_forecast(self):
        c = dlm_uni_df(
                F=lego.E(2), G=lego.J(2), delta=.95)
        n = 30
        nAhead = 5
        y = np.array(range(n)) + np.random.normal(0, .1, n)
        init = param.uni_df(
                m=np.asmatrix(np.zeros((2,1))), 
                C=np.eye(2))
        filt = c.filter(y, init)
        fc = c.forecast(filt, nAhead)
        print
        print 'at n:    ' + (y[-1] + 5).__str__()
        print 'predict: ' + fc['f'][-1].__str__()

    def test_dlm_uni_forecast(self):
        dlm = dlm_uni(F=lego.E(2), G=lego.J(2), discount=.95)
        print dlm
        n = 30
        nAhead = 5
        y = np.array(range(n)) + np.random.normal(0, .1, n)
        init = param.uni_df(
                m=np.asmatrix(np.zeros((2,1))), 
                C=np.eye(2))
        filt = dlm.filter(y, init)
        fc = dlm.forecast(filt, nAhead)
        print
        print 'at n:    ' + (y[-1] + 5).__str__()
        print 'predict: ' + fc['f'][-1].__str__()

    def test_dlm_uni_forecast2(self):
        dlm = dlm_mod.poly(1, discount=.95, V=1) + dlm_mod.arma([.95,.04], V=1)
        print dlm
        n = 30
        nAhead = 5
        y = np.array(range(n)) + np.random.normal(0, 1, n)
        init = param.uni_df(
                m=np.asmatrix(np.zeros((dlm.p,1))), 
                C=np.eye(dlm.p))
        filt = dlm.filter(y, init)
        fc = dlm.forecast(filt, nAhead)

        future_idx = np.linspace(n, n+nAhead, nAhead)
        idx = np.arange(n)
        
        theta = dlm.ffbs(y, init)

        #plt.figure()
        #plt.scatter(idx, y, color='grey')
        #plt.plot(idx, map(lambda f: f.f, filt), color='red')
        #plt.plot(future_idx, fc['f'], color='red')
        #plt.plot(idx, sm_f, color='blue')
        #plt.show()

        
        #print
        #print 'hi'
        #print 'at n:    ' + (y[-1] + 5).__str__()
        #print 'predict: ' + fc['f'][-1].__str__()

        #print 'smooth a: ', sm['a']
        #print 'smooth R: ', sm['R']

