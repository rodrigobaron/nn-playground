import numpy as np
from .data_iterator import DataIterator

class BatchIterator(DataIterator):
    """TODO: BatchIterator docs"""
    def __init__(self, batch_size, shuffle=False):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield batch_inputs, batch_targets

