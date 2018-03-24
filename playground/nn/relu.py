from .activation import Activation
import numpy as np

class ReLU(Activation):
    """TODO: Relu docs"""

    def __init__(self):
        super().__init__()

    def f(self, inputs):
        return np.maximum(inputs, 0)

    def f_prime(self, inputs):
        ret = inputs.copy()
        ret[inputs < 0] = 0
        return ret