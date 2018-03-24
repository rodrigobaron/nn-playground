from .activation import Activation
import numpy as np

class LeakyReLU(Activation):
    """TODO: LeakyReLU docs"""

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def f(self, inputs):
        return np.maximum(inputs * self.negative_slope, inputs)

    def f_prime(self, inputs):
        ret = inputs.copy()
        ret[inputs < 0] *= self.negative_slope
        return ret