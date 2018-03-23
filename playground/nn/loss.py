
class Loss:
    """TODO: loss docs"""
    def loss(self, predicted, truth):
        raise NotImplementedError

    def grad(self, predicted, truth):
        raise NotImplementedError

    def __call__(self, predicted, truth):
        return self.loss(predicted, truth)