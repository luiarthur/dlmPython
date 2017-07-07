from .dlm import dlm
from .lego import join
from .param import uni as param_uni
from scipy.linalg import block_diag
from scipy.stats import t as t_dist
from scipy.stats import norm
import numpy as np
from numpy.linalg import inv

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class dlm_uni(dlm):

    def __init__(self, F, G, V=None, W=None, discount=None):

        if discount is not None:
            if type(discount) in [int, float]:
               assert 0 < discount <= 1, "discount needs to be in unit interval"

        self.discount = discount
        self.F = F
        self.G = G
        self.V = V
        p = G.shape[0]
        self.p = p

        if W is None and discount is None:
            self.W = np.zeros( (p,p) )
        elif W is not None:
            self.W = W
        elif discount is not None:
            self.W = np.zeros( (p,p) ) + np.nan

        self.__dimension__ = p
        self.__num_components__ = 1


    def __add__(self, other):

        assert (self.V is None) - (other.V is None) == 0, "Both dlm's must either have known V or unknown V."

        F = np.concatenate( (self.F, other.F) )
        G = block_diag(self.G, other.G)
        V = None if self.V is None else self.V + other.V
        W = block_diag(self.W, other.W)
        discount = join(self.discount, other.discount)

        new_dlm = dlm_uni(F=F, G=G, V=V, W=W, discount=discount)
        new_dlm.__dimension__ = join(self.__dimension__, other.__dimension__)
        new_dlm.__num_components__ = self.__num_components__ + other.__num_components__

        return new_dlm

    def __str__(self):
        return "F:\n" + self.F.__str__() + "\n\n" + \
               "G:\n" + self.G.__str__() + "\n\n" + \
               "V:\n" + self.V.__str__() + "\n\n" + \
               "W:\n" + self.W.__str__() + "\n\n" + \
               "discount:\n" + self.discount.__str__() + "\n\n" + \
               "num components:\n" + self.__num_components__.__str__() + "\n\n" + \
               "dimension:\n" + self.__dimension__.__str__()


    def __get_block__(self, M, i):
        cum_dim = np.insert(np.cumsum(self.__dimension__), 0, 0)
        dim_lower = cum_dim[:-1]
        dim_upper = cum_dim[1:]

        return M[dim_lower[i]:dim_upper[i],
                 dim_lower[i]:dim_upper[i]] 

    def __compute_W__(self, prev_C):
        W_list = [None] * self.__num_components__
        d = self.discount if self.__num_components__ > 1 else [self.discount]

        ### See Section 6.3.2 (Component discounting) of W&H.
        for i in range(self.__num_components__):
            Gi = self.__get_block__(self.G, i)
            prev_Ci = self.__get_block__(prev_C, i)

            if d[i] is None:
                W_list[i] = self.__get_block__(self.W, i)
            else:
                W_list[i] = (1-d[i]) / d[i] * Gi * prev_Ci * Gi.T

        return reduce(lambda a,b: block_diag(a,b), W_list, np.eye(0))

    def filter(self, y, init):
        N = len(y)
        out = [init]*N
        G = self.G
        Gt = G.T
        Ft = np.asmatrix(self.F)
        F = Ft.T

        for i in xrange(N):
            prev = out[i-1] if i > 0 else init
            n = prev.n + 1
            W = self.__compute_W__(prev.C)
            a = G * prev.m
            R = G * prev.C * Gt + W
            f = np.asscalar(Ft * a)
            Q = np.asscalar(Ft * R * F) + prev.S
            e = y[i] - f
            S = (prev.S + prev.S / n * (e*e / Q - 1)) if self.V is None else self.V
            A = R * F / Q
            m = a + A*e
            C = S / prev.S * (R - A*A.T * Q)
            if not is_pos_def(C):
                C += np.eye(self.p) * 1E-10

            if Q < 0:
                print "i:"
                print i
                print "W:"
                print W
                print "R:"
                print R
                print "S_prev:"
                print prev.S
                print "S:"
                print S
                print "Q:"
                print Q
                print "C:"
                print C
                print "discount:"
                print self.discount

            assert Q >= 0, "Q cannot be negative!"

            out[i] = param_uni(m=m,C=C,a=a,R=R,f=f,Q=Q,n=n,S=S)

        return out

    def forecast(self, filt, nAhead=1, linear_decay=False):
        last_param = filt[-1]
        init = (last_param.m, last_param.C, 
                last_param.f, last_param.Q)

        G = self.G
        Gt = G.T
        Ft = np.asmatrix(self.F)
        F = Ft.T
        W = self.__compute_W__(last_param.C)

        out = [None] * nAhead
        # TODO: Try Corollary 4.1, 4.2 of W&H
        for i in xrange(nAhead):
            prev = out[i-1] if i > 0 else init
            (prev_a, prev_R, prev_f, prev_Q) = prev
            a = G * prev_a

            # See W&H 6.3.3 (Practical discounting strategy for k-step ahead forecasts)
            if not linear_decay:
                # Exponential decay of information
                W = self.__compute_W__(prev_R)

            R = G * prev_R * Gt + W
            f = Ft * a
            Q = Ft * R * F + last_param.S
            out[i] = (a, R, f, Q)
            
        out_f = map(lambda x: np.asscalar(x[2]), out)
        out_Q = map(lambda x: np.asscalar(x[3]), out)
        return {'f': out_f, 'Q': out_Q, 'n': last_param.n}

    def get_ci(self, f, Q, n, alpha=.05):
        assert len(f) == len(Q), "required: len(f) == len(Q)"

        len_f = len(f)
        lower = [None] * len_f
        upper = [None] * len_f
        num_std = norm.ppf(1 - alpha/2)

        for i in range(len_f):

            if self.V is None:
                num_std = t_dist(df=n[i]-1).ppf(1 - alpha/2)

            lower[i] = f[i] - np.sqrt(Q[i]) * num_std
            upper[i] = f[i] + np.sqrt(Q[i]) * num_std

        return {'lower': lower, 'upper': upper}

    def draw_forecast(self, f, Q, n, num_draws):
        """
        f:         forecast mean parameter (float)
        Q:         forecast squared scale parameter (float)
        n:         number of samples trained from so far (int)
        num_draws: number of draws (int)
        """
        if self.V is None:
            return t_dist(df=n-1).rvs(num_draws) * np.sqrt(Q) + f
        else:
            return np.random.randn(num_draws) * np.sqrt(Q) + f

    # TODO:
    def ffbs(self, y, init):
        """
        ffbs (Forward filtering Backwards Sampling)
        """
        assert self.V is not None, "ffbs can only be used for conditionally Normal models!"

        filt = self.filter(y, init)
        p = self.p
        N = len(y)
        theta = np.zeros((N, p))

        for t in reversed( xrange(N) ):

            if t == N-1:
                ht = np.asarray(filt[t].m).flatten()
                Ht = filt[t].C
                theta[t,:] = np.random.multivariate_normal(ht, Ht)
            else:
                Ct = filt[t].C
                Gt_next = self.G
                Rt_next = filt[t+1].R
                Bt = Ct * Gt_next.T * inv(Rt_next)
                at_next = filt[t+1].a
                ht = np.asarray(filt[t].m + Bt * Rt_next * (np.asmatrix(theta[t+1, :]).T - at_next)).flatten()
                Ht = Ct - Bt * Rt_next * Bt.T
                theta[t, :] = np.random.multivariate_normal(ht, Ht)

        return theta
