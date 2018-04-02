import numpy as np
from .optimizer import Optimizer

class RMSprop(Optimizer):
    """TODO: RMSprop docs"""
    def __init__(self, lr, momentum=0.9):
        super().__init__()

        self.lr = lr
        self.momentum = momentum

        self.cache = {}

    def step(self, model):
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.cache:
                self.cache.update({i: np.zeros(grad.shape)})
            self.cache[i] = (self.momentum * self.cache[i] + (1.0 - self.momentum) * grad**2)
            param -= self.lr * grad / (np.sqrt(self.cache[i]) + 1e-8)