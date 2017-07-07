import numpy as np
from .lego import E, J, join
from .dlm_uni import dlm_uni


def arma(ar=[], ma=[], tau2=1, V=None):
    """
    ARMA component for DLM
    - ar: list of ar coefficients
    - ma: list of ma coefficients
    - tau2: variance for evolution matrix
    - V: variance for observations

    Note that there is no discount option here because W is assumed to be 
    a matrix of zeros but with W_{1,1} = sig2

    see West & Prado (p.75)
    """
    assert not(ar is [] and ma is []), "You must specify at least one of 'ar' or 'ma'!"
    p = len(ar)
    q = len(ma)
    m = max(p, q+1)
    
    phi = join(np.array(ar), np.zeros(m-p))
    rest = np.vstack( (np.eye(m-1), np.zeros(m-1)) )
    G = np.column_stack( (phi, rest) )

    psi = join(np.array(ma), np.zeros(m-1-q))
    omega = np.asmatrix(join(1, psi)).transpose()
    W = tau2 * omega * omega.transpose()
    
    return dlm_uni(F=E(m), G=G, V=V, W=W)

def poly(order=1, V=None, W=None, discount=None):
    """
    Polynomial trend component for DLM
    - order: order 0 polynomial => mean forecast function (random walk model)
             order 1 polynomial => linear forecast function
             order 2 polynomial => quadratic forecast function
    """
    assert order >= 0, "order needs to be > 0"
    p = order + 1
    return dlm_uni(F=E(p), G=J(p), V=V, W=W, discount=discount)

def seasonal():
    return NotImplemented

def reg():
    """
    Creates dlm model for regression
    """
    return NotImplemented


