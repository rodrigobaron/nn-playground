from .optimizer import Optimizer

class SGD(Optimizer):
    """TODO: SGD docs"""
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

    def step(self, model):
        for param, grad in model.parameters():
            param -= self.lr * grad