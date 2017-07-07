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

    def test_dlm_uni_forecast3(self):
        def inv_logit(x):
            return 1 / (1 + np.exp(-x))

        #dlm = dlm_mod.poly(1, discount=.5)
        dlm = dlm_mod.poly(1, discount=.99)
        print dlm

        n = 6984
        nAhead = 1000

        #y = inv_logit(np.linspace(-10, 10, n)) + np.random.normal(0, .01, n)
        y = inv_logit(np.linspace(-10, 10, n)) * .1

        plt.plot(y)
        plt.show()

        init = param.uni(
                m=np.asmatrix(np.zeros((dlm.p,1))), 
                C=np.eye(dlm.p))
        filt = dlm.filter(y, init)

        one_step_f = map(lambda x: x.f, filt)
        one_step_Q = map(lambda x: x.Q, filt)
        one_step_n = map(lambda x: x.n, filt)
        # credible intervals for predictions
        ci_one_step = dlm.get_ci(one_step_f, one_step_Q, one_step_n)

        fc = dlm.forecast(filt, nAhead, linear_decay=True)
        #ci = dlm.get_ci(fc['f'], [fc['Q'][0]]*nAhead, [fc['n']]*nAhead)
        ci = dlm.get_ci(fc['f'], fc['Q'], [fc['n']]*nAhead)

        future_idx = np.linspace(n, n+nAhead, nAhead)
        idx = np.arange(n)
        
        ### PLOT RESULT
        plt.fill_between(range(n), ci_one_step['lower'], ci_one_step['upper'], color='lightblue')
        plt.fill_between(future_idx, ci['lower'], ci['upper'], color='pink')

        #plt.plot(range(n), y[:n], color='grey', label='Data')
        plt.scatter(range(n), y[:n], color='grey', label='Data', s=15)
        plt.plot(future_idx, fc['f'], 'r--', label='Forecast')
        plt.plot(range(n), one_step_f, 'b--', label='One-step-Ahead')

        plt.xlabel('time')
        #legend = plt.legend(loc='lower right')
        plt.ylim([
          min(ci['lower'][100:]+ci_one_step['lower'][100:]),
          max(ci['upper'][100:]+ci_one_step['upper'][100:])
          ])
        plt.show()


