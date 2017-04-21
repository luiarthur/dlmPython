from abc import ABCMeta, abstractmethod
import numpy as np

class dlm:
    __metaclass__ = ABCMeta

    def __init__(self, F, G, V, W):
        self.F = F
        self.G = G
        self.V = V
        self.W = W
    
    # Requres implementation for instantiation of derived classes
    @abstractmethod
    def filter(self):
        return NotImplemented

    # Requres implementation for instantiation of derived classes
    @abstractmethod
    def forecast(self):
        return NotImplemented

    def smooth(self):
        return NotImplemented

    def backSample(self):
        return NotImplemented
