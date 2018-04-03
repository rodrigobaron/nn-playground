import numpy as np
from .layer import Layer

class Reshape(Layer):
    """TODO: Reshape layer doc"""
    def __init__(self, *out_shape):
        super().__init__()
        self.out_shape = (-1,) + out_shape

    def forward(self, inputs):
        self.in_shape = inputs.shape
        return np.reshape(inputs, self.out_shape)

    def backward(self, grad):
        return np.reshape(grad, self.in_shape)
