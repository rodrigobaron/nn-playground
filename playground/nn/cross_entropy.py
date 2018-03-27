import numpy as np
from .loss import Loss
from ..utils.softmax import Softmax

class CrossEntropy(Loss):
    """TODO: CrossEntropy docs"""
    def __init__(self, regularization=None):
        self.regularization = regularization
        self.softmax = Softmax()

    def loss(self, predicted, truth):
        m = predicted.shape[0]
        p = self.softmax(predicted)
        llik = -np.log(p[range(m), np.argmax(truth)])

        _loss = np.sum(llik) / m

        if self.regularization is None:
            return _loss

        return _loss + self.regularization()

    def grad(self, predicted, truth):
        m = predicted.shape[0]
        grad_y = self.softmax(predicted)
        grad_y[range(m), np.argmax(truth)] -= 1.
        grad_y /= m

        return grad_y