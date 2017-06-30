from .dlm import dlm
from .lego import join
from .param import uni as param_uni
from scipy.linalg import block_diag
from scipy.stats import t as t_dist
import numpy as np

class dlm_uni(dlm):

    def __init__(self, F, G, V=None, W=None, discount=None):

        if discount is not None:
            if type(discount) in [int, float]:
               assert 0 < discount <= 1, "discount needs to be in unit interval"

        self.discount = np.array(discount)
        self.F = F
        self.G = G
        self.V = V
        p = G.shape[0]

        if W is None and discount is None:
            self.W = np.zeros( (p,p) )
        elif W is not None:
            self.W = W
        elif discount is not None:
            self.W = np.zeros( (p,p) ) + np.nan

        self.__dimension__ = np.array(p)
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

    def __compute_W_(self, prev_C):
        return NotImplemented

    def filter(self):
        pass

    def forecast(self):
        pass

    def smooth(self):
        pass

    def back_sample(self):
        pass


