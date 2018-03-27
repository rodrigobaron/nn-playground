from .activation import Activation
import numpy as np

class Sigmoid(Activation):
    """TODO: Sigmoid docs"""

    def __init__(self):
        super().__init__()

    def f(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def f_prime(self, inputs):
        return self.f(inputs) * (1 - self.f(inputs))