from .dlm import dlm
from .lego import join
from .param import uni_df as param_uni_df
from scipy.linalg import block_diag
from scipy.stats import t as t_dist
import numpy as np

class dlm_uni_df(dlm):

    def __init__(self, F, G, dim=None, delta=0.95):
        p = G.shape[0] if dim is None else dim
        self.F = F
        self.G = G
        self.dim = p
        self.delta = delta
        num = [int, float]
        self.num = num
        self.num_components = 1 if type(p) in num else len(p)
        #
        cum_dim = np.insert(np.cumsum(self.dim), 0, 0)
        self.cum_dim = cum_dim
        self.dim_lower = cum_dim[:-1]
        self.dim_upper = cum_dim[1:]
        # Discount Factor 
        delta_arr = delta if type(delta) is np.ndarray else np.array([delta])
        self.df = map(lambda d: (1-d) / d, delta_arr)

    def __str__(self):
        return "F:\n" + self.F.__str__() + "\n\n" + \
               "G:\n" + self.G.__str__() + "\n\n" + \
               "delta:\n" + self.delta.__str__() + "\n\n" + \
               "dim:\n" + self.dim.__str__()

    def __add__(self, other):
        F = np.concatenate( (self.F, other.F) )
        G = block_diag(self.G, other.G)
        dim = join(self.dim, other.dim)
        delta = join(self.delta, other.delta)
        return dlm_uni_df(F=F, G=G, dim=dim, delta=delta)


    # Compute W matrix based on previous C matrix
    def __compute_W__(self, prev_C):
        def block(M, i):
            return M[self.dim_lower[i]:self.dim_upper[i],
                     self.dim_lower[i]:self.dim_upper[i]] 

        if self.num_components > 1:
            W_list = [None] * self.num_components
            ###        see section 6.3.2 (Component discounting)
            ###        of W&H.
            for i in range(self.num_components):
                Gi = block(self.G, i)
                prev_Ci = block(prev_C, i)
                #Gi = self.G[self.dim_lower[i]:self.dim_upper[i],
                #            self.dim_lower[i]:self.dim_upper[i]]
                #prev_Ci = prev_C[self.dim_lower[i]:self.dim_upper[i],
                #                 self.dim_lower[i]:self.dim_upper[i]]
                W_list[i] = self.df[i] * Gi * prev_Ci * Gi.transpose()

            print Gi
            print prev_Ci
            print reduce(lambda a,b: block_diag(a,b), W_list, np.eye(0))

            return reduce(lambda a,b: block_diag(a,b), W_list, np.eye(0))
        else:
            return self.df[0] * self.G * prev_C * self.G.transpose()

    ### FIXME ###
    def filter(self, y, init):
        #W = self.__compute_W__(np.eye(self.G.shape[0]))
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
            S = prev.S + prev.S / n * (e*e / Q - 1)
            A = R * F / Q
            m = a + A*e
            C = S / prev.S * (R - A*A.transpose() * Q)

            out[i] = param_uni_df(m=m,C=C,a=a,R=R,f=f,Q=Q,n=n,S=S)

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

        lower = [f[i] + np.sqrt(Q[i]) * t_dist(df=n[i]-1).ppf(alpha/2) 
                 for i in range(len_f)]
        upper = [f[i] + np.sqrt(Q[i]) * t_dist(df=n[i]-1).ppf(1 - alpha/2) 
                 for i in range(len_f)]

        return {'lower': lower, 'upper': upper}

    def draw_forecast(self, f, Q, n, num_draws):
        """
        f:         forecast mean parameter (float)
        Q:         forecast scale parameter (float)
        n:         number of samples trained from so far (int)
        num_draws: number of draws (int)
        """
        return t_dist(df=n-1).rvs(num_draws) * np.sqrt(Q) + f

