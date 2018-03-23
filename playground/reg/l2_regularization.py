import numpy as np

from .regularization import Regularization


class L2Regularization(Regularization):
    """TODO: L2Regularization docs"""
    def __init__(self, model, lam=1e-3):
        super().__init__(model)

        self.lam = lam

    def forward(self):
        return np.sum([.5 * self.lam * np.sum(layer.params['w'] * layer.params['w'])
                       for layer in self.model.layers
                       if 'w' in layer.params])

    def backward(self):
        return np.sum([self.lam * layer.params['w']
                       for layer in self.model.layers
                       if 'w' in layer.params])
