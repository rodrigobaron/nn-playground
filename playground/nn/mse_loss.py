import numpy as np
from .loss import Loss

class MSELoss(Loss):
    """TODO: MSELoss docs"""
    def __init__(self, regularization=None):
        self.regularization = regularization

    def loss(self, predicted, truth):
        m = predicted.shape[0]
        _loss = .5 * np.sum((predicted - truth)**2) / m

        if self.regularization is None:
            return _loss

        return _loss + self.regularization()

    def grad(self, predicted, truth):
        m = predicted.shape[0]
        grad = 2 * (predicted - truth)
        return grad / m