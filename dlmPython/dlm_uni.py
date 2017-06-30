from .dlm import dlm
from .lego import join
from .param import uni as param_uni
from scipy.linalg import block_diag
from scipy.stats import t as t_dist
from scipy.stats import norm
import numpy as np

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
                W_list[i] = (1-d[i]) / d[i] * Gi * prev_Ci * Gi.transpose()

        return reduce(lambda a,b: block_diag(a,b), W_list, np.eye(0))

    def filter(self, y, init):
        N = len(y)
        out = [init]*N
        G = self.G
        Gt = G.transpose()
        Ft = np.asmatrix(self.F)
        F = Ft.transpose()

        for i in xrange(N):
            prev = out[i-1] if i > 0 else init
            n = prev.n + 1
            W = self.__compute_W__(prev.C)
            a = G * prev.m
            R = G * prev.C * Gt + W
            f = (Ft * a)[0,0]
            Q = (Ft * R * F + prev.S)[0,0]
            e = y[i] - f
            S = prev.S + prev.S / n * (e*e / Q - 1) if self.V is None else self.V
            A = R * F / Q
            m = a + A*e
            C = S / prev.S * (R - A*A.transpose() * Q)

            out[i] = param_uni(m=m,C=C,a=a,R=R,f=f,Q=Q,n=n,S=S)

        return out

    def forecast(self, filt, nAhead=1, linear_decay=False):
        last_param = filt[-1]
        init = (last_param.m, last_param.C, 
                last_param.f, last_param.Q)

        G = self.G
        Gt = G.transpose()
        Ft = np.asmatrix(self.F)
        F = Ft.transpose()
        W = None

        out = [None] * nAhead
        for i in xrange(nAhead):
            prev = out[i-1] if i > 0 else init
            (prev_a, prev_R, prev_f, prev_Q) = prev
            a = G * prev_a

            # See W&H 6.3.3 (Practical discounting strategy for k-step ahead forecasts)
            if linear_decay:
                if W is None:
                    # linear decay of information
                    W = self.__compute_W__(prev_R)
            else:
                # Exponential decay of information
                W = self.__compute_W__(prev_R)

            R = G * prev_R * Gt + W
            f = Ft * a
            Q = Ft * R * F + last_param.S
            out[i] = (a, R, f, Q)
            
        out_f = map(lambda x: x[2][0,0], out)
        out_Q = map(lambda x: x[3][0,0], out)
        return {'f': out_f, 'Q': out_Q, 'n': last_param.n}

    def get_ci(self, f, Q, n, alpha=.05):
        assert len(f) == len(Q), "required: len(f) == len(Q)"

        len_f = len(f)

        num_std = t_dist(df=n[i]-1).ppf(1 - alpha/2) if self.V is None else norm.ppf(1 - alpha/2)

        lower = [f[i] - np.sqrt(Q[i]) * num_std for i in range(len_f)]
        upper = [f[i] + np.sqrt(Q[i]) * num_std for i in range(len_f)]

        return {'lower': lower, 'upper': upper}

    def draw_forecast(self, f, Q, n, num_draws):
        """
        f:         forecast mean parameter (float)
        Q:         forecast scale parameter (float)
        n:         number of samples trained from so far (int)
        num_draws: number of draws (int)
        """
        if self.V is None:
            return t_dist(df=n-1).rvs(num_draws) * np.sqrt(Q) + f
        else:
            return np.random.randn(num_draws) * np.sqrt(Q) + f

    def smooth(self):
        pass

    def back_sample(self):
        pass


