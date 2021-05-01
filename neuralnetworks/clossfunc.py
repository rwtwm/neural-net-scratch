from abc import ABC, abstractmethod
import numpy as np


class LossFunction:
    @abstractmethod
    def loss(self, values, expected):
        '''Compute the loss of the computed vs expected values'''
        pass

    @abstractmethod
    def dloss(self, values, expected):
        '''derivative of the loss function'''
        pass


class MSELoss(LossFunction):
    '''Mean squared error loss function'''
    def loss(self, values, expected):
        return np.mean((values - expected) ** 2)

    def dloss(self, values, expected):
        return 2 * (values - expected)/values.size


