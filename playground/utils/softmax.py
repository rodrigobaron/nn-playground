import numpy as np

class Softmax:
    """TODO: Softmax docs"""
    def forward(self, inputs):
        ex = np.exp((inputs.T - np.max(inputs)).T)
        return (ex.T / ex.sum()).T

    def __call__(self, inputs):
        return self.forward(inputs)