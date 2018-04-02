import numpy as np
from .layer import Layer

class Dropout(Layer):
    """TODO: Dropout layer doc"""
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def forward(self, inputs):
        self.rb = np.random.binomial(1, self.p, size=inputs.shape) / self.p
        print('forward', self.rb)
        return inputs * self.rb

    def backward(self, grad):
        print('backward', grad * self.rb)
        return grad * self.rb