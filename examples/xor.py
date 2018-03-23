import sys
sys.path.append(sys.path[0] + "/..")

import numpy as np

from playground.nn import Model, MSELoss
from playground.nn import Linear, Tanh
from playground.utils import BatchIterator
from playground.optim import SGD
from playground.reg import L2Regularization

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

model = Model([
Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
], clipping=6.)

iterator = BatchIterator(batch_size=4, shuffle=True)
regularization = L2Regularization(model)
loss = MSELoss(regularization=regularization)

optimizer = SGD(lr=0.01)
epochs = 5000

for i, epoch in enumerate(range(epochs)):
    epoch_loss = 0.0
    for batch_inputs, batch_targets in iterator(inputs, targets):
        predicted = model(batch_inputs)
        epoch_loss += loss(predicted, batch_targets)
        grad = loss.grad(predicted, batch_targets)
        model.backward(grad)
        optimizer.step(model)
    if i % 100 == 99:
        print(epoch, epoch_loss)

for x, y in zip(inputs, targets):
    predicted = model(x)
    print(x, np.array(np.round(predicted), dtype=int), y)