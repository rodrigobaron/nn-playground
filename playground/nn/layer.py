
class Layer:
    """ TODO: layer doc"""
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs)