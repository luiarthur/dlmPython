from .dlm import dlm
from .lego import join
from .param import uni_df as param_uni_df
from scipy.linalg import block_diag
import numpy as np

class dlm_uni_df(dlm):

    def __init__(self, F, G, V, dim=None, delta=0.95):
        p = G.shape[0] if dim is None else dim
        self.F = F
        self.G = G
        self.V = V
        self.dim = p
        self.delta = delta
        num = [int, float]
        self.num = num
        self.num_components = 1 if type(p) in num else len(p)
        #
        cum_dim = np.insert(np.cumsum(dim), 0, 0)
        self.cum_dim = cum_dim
        self.dim_lower = cum_dim[:-1]
        self.dim_upper = cum_dim[1:]
        # Discount Factor 
        delta_arr = delta if type(delta) is np.ndarray else np.array([delta])
        self.df = map(lambda d: (1-d) / d, delta_arr)

    def __str__(self):
        return "F:\n" + self.F.__str__() + "\n\n" + \
               "G:\n" + self.G.__str__() + "\n\n" + \
               "V:\n" + self.V.__str__() + "\n\n" + \
               "delta:\n" + self.delta.__str__() + "\n\n" + \
               "dim:\n" + self.dim.__str__()

    def __add__(self, other):
        F = np.concatenate( (self.F, other.F) )
        G = block_diag(self.G, other.G)
        V = self.V + other.V
        dim = join(self.dim, other.dim)
        delta = join(self.delta, other.delta)
        return dlm_uni_df(F=F, G=G, V=V, dim=dim, delta=delta)

    # Compute W matrix based on previous C matrix
    def __compute_W__(self, prev_C):
        W_list = [None] * self.num_components

        for i in xrange(self.num_components):
            Gi = self.G[self.dim_lower[i]:self.dim_upper[i],
                        self.dim_lower[i]:self.dim_upper[i]]
            prev_Ci = prev_C[self.dim_lower[i]:self.dim_upper[i],
                             self.dim_lower[i]:self.dim_upper[i]]
            W_list[i] = self.df[i] * Gi * prev_Ci * Gi.transpose()

        return reduce(lambda a,b: block_diag(a,b), 
                W_list, np.eye(0))

    ### FIXME ###
    def filter(self, y, init):
        W = self.__compute_W__(np.eye(self.G.shape[0]))
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
            S = prev.S + prev.S / n * (e*e/Q-1)
            A = R * self.F / Q
            m = a + A*e
            C = S / prev.S * (R - A*A.transpose()*Q)

            out[i] = param_uni_df(m=m,C=C,a=a,R=R,f=f,Q=Q,n=n,S=S)

        return out

    def forecast():
        pass