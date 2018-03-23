from .layer import Layer

class Activation(Layer):
    """TODO: Activation docs"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        return self.f_prime(self.inputs) * grad

    def f(self, inputs):
        raise NotImplementedError

    def f_prime(self, inputs):
        raise NotImplementedError