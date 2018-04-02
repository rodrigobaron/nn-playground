import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    """TODO: Adagrad docs"""
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

        self.cache = {}

    def step(self, model):
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.cache:
                self.cache.update({i: np.zeros(grad.shape)})
            self.cache[i] = + grad**2
            param -= self.lr * grad / (np.sqrt(self.cache[i]) + 1e-8)