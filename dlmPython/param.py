import numpy as np

# Parameter for univariate DLM with discount factors
class uni_df:
    def __init__(self, m, C, a=None, R=None, f=None, Q=None, n=1, S=1):
        self.a = a
        self.R = R
        self.f = f
        self.Q = Q
        self.m = m
        self.C = C
        self.n = n
        self.S = S

class uni:
    def __init__(self, m, C, a=None, R=None, f=None, Q=None, n=1, S=1):
        self.a = a
        self.R = R
        self.f = f
        self.Q = Q
        self.m = m
        self.C = C
        self.n = n
        self.S = S

