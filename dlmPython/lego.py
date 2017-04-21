import numpy as np


def E2(p):
    """
    Creates a vector of length 'p' with the first
    element being '1' and the other elements being '0'
    """
    out = np.zeros(p)
    out[0] = 1
    return out


def J(p, lam=1):
    """
    Creates a jordan matrix of dimension 'p'. 
    The default value for the diagonal is '1'.
    """
    M = np.eye(p) * lam
    i, j = np.indices(M.shape)
    M[i == j-1] = 1
    return M


def join(a, b):
    num = [int, float]
    if type(a) == type(b) == np.ndarray:
        return np.concatenate((a, b), axis=0)
    elif type(a) in num and type(b) in num:
        return np.array([a, b])
    elif type(a) in num and type(b) == np.ndarray:
        return np.insert(b, 0, a)
    elif type(a) == np.ndarray and type(b) in num:
        return np.append(a, b)
    else:
        assert False, "type of a, b needs to be " + \
                      "in [int, float, np.ndarray]" 


