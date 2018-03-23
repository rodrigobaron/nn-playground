import numpy as np

class Model:
    def __init__(self, layers = [], clipping=None):
        self.layers = layers
        self.clipping = clipping

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        for layer in self.layers:
            for key, param in layer.params.items():
                grad = layer.grads[key]
                if self.clipping is not None:
                    grad = np.clip(grad, -self.clipping, self.clipping)
                yield param, grad

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs)