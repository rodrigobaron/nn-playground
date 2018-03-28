import numpy as np
from .loss import Loss
from ..utils.softmax import Softmax

class BinaryCrossEntropy(Loss):
    """TODO: BinaryCrossEntropy docs"""
    def __init__(self, regularization=None):
        self.regularization = regularization
        self.softmax = Softmax()

    def loss(self, predicted, truth):
        m = predicted.shape[0]

        _loss = -np.mean(truth * np.log2(predicted + 1e-8)+ (1 - truth) * np.log2((1-predicted) + 1e-8)) / m

        if self.regularization is None:
            return _loss

        return _loss + self.regularization()

    def grad(self, predicted, truth):
        m = predicted.shape[0]
        grad = truth / (predicted + 1e-8) - (1 - truth) / (1 - predicted + 1e-8)
        grad *= (predicted.size * np.log(2).astype(np.float32))

        return -grad / m