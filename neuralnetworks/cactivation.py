from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction:
    '''Abstract class for defining activation functions'''

    @abstractmethod
    def f(self, x):
        '''The method that implements the function'''
        pass

    @abstractmethod
    def df(self, x):
        '''The method that implements the derivative of the function'''
        pass


class LeakyReLU(ActivationFunction):
    '''Leaky Rectified Linear Unit - See structure for a little more on this'''
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def f(self, x):
        return np.maximum(x, x * self.alpha)

    def df(self, x):
        return np.maximum(x > 0, self.alpha)