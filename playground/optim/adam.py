import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    """TODO: Adam docs"""
    def __init__(self, lr, momentum1=0.9, momentum2=0.999):
        super().__init__()

        self.lr = lr
        self.momentum1 = momentum1
        self.momentum2 = momentum2

        self.cache1 = {}
        self.cache2 = {}

    def step(self, model):
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.cache1:
                self.cache1.update({i: np.zeros(grad.shape)})
            if i not in self.cache2:
                self.cache2.update({i: np.zeros(grad.shape)})

            self.cache1[i] = (self.momentum1 * self.cache1[i] + (1.0 - self.momentum1) * grad)
            self.cache2[i] = (self.momentum2 * self.cache2[i] + (1.0 - self.momentum2) * grad**2)
            param -= self.lr * self.cache1[i] / (np.sqrt(self.cache2[i]) + 1e-8)