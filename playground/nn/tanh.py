from .activation import Activation
import numpy as np

class Tanh(Activation):
    """TODO: Tanh docs"""

    def __init__(self):
        super().__init__()

    def f(self, inputs):
        return np.tanh(inputs)

    def f_prime(self, inputs):
        return 1 - self.f(inputs) ** 2