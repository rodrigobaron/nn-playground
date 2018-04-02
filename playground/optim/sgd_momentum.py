import numpy as np
from .optimizer import Optimizer

class SGDMomentum(Optimizer):
    """TODO: SGDMomentum docs"""
    def __init__(self, lr, momentum=0.9):
        super().__init__()

        self.lr = lr
        self.momentum = momentum

        self.velocity = {}

    def step(self, model):
        for i, (param, grad) in enumerate(model.parameters()):
            if i not in self.velocity:
                self.velocity.update({i: np.zeros(grad.shape)})
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * grad
            param -= self.velocity[i]