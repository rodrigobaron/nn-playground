import numpy as np

from .regularization import Regularization


class L1Regularization(Regularization):
    """TODO: L1Regularization docs"""
    def __init__(self, model, lam=1e-3):
        super().__init__(model)

        self.lam = lam

    def forward(self):
        return np.sum([self.lam * np.abs(layer.params['w'])
                       for layer in self.model.layers
                       if 'w' in layer.params])

    def backward(self):
        return np.sum([self.lam * layer.params['w'] / (np.abs(layer.params['w']) + 1e-8)
                       for layer in self.model.layers
                       if 'w' in layer.params])
