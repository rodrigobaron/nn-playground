import numpy as np
from .loss import Loss
from ..utils.softmax import Softmax

class CrossEntropyLoss(Loss):
    """TODO: CrossEntropyLoss docs"""
    def __init__(self, regularization=None):
        self.regularization = regularization
        self.softmax = Softmax()

    def loss(self, predicted, truth):
        m = predicted.shape[0]

        prob = self.softmax(predicted)
        llike = -np.log(prob[range(m), truth])

        _loss = np.sum(llike) / m

        if self.regularization is None:
            return _loss

        return _loss + self.regularization()

    def grad(self, predicted, truth):
        m = predicted.shape[0]
        prob = self.softmax(predicted)
        prob[range(m), truth] -= 1.

        return prob / m