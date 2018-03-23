
class Regularization:
    """TODO: Regularization docs"""
    def __init__(self, model):
        self.model = model

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self):
        return self.forward()
