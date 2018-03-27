import sys

sys.path.append(sys.path[0] + "/..")

import numpy as np

from playground.nn import Model, MSELoss, CrossEntropy
from playground.nn import Linear, Tanh
from playground.utils import BatchIterator
from playground.optim import SGD
from playground.reg import L2Regularization
#
# from playground.data.mnist import load_mnist
#
# x_train, y_train = load_mnist(train=True, one_hot=True)
# x_test, y_test = load_mnist(train=False, one_hot=True)
#
img_size = 28
num_classes = 10
img_size_flat = img_size * img_size

#
# print('shape x', x_train.shape)
# print('shape y', y_train.shape)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=False)



# x_train = x_train.reshape(-1, img_size_flat)
# x_test = x_test.reshape(-1, img_size_flat)
#
# print('test', y_test[0])

# print('shape x', x_train.shape)

model = Model([
    Linear(input_size=img_size_flat, output_size=num_classes)
])


iterator = BatchIterator(batch_size=100, shuffle=True)
regularization = L2Regularization(model)
loss = CrossEntropy()

optimizer = SGD(lr=0.0001)
epochs = 5000

for i, epoch in enumerate(range(epochs)):
    epoch_loss = 0.0
    # for batch_inputs, batch_targets in iterator(x_train, y_train):
    x_batch, y_true_batch = data.train.next_batch(100)

    predicted = model(x_batch)
    epoch_loss += loss(predicted, y_true_batch)
    grad = loss.grad(predicted, y_true_batch)
    model.backward(grad)
    optimizer.step(model)
    if i % 100 == 99:
        print(epoch, epoch_loss)

# for x, y in zip(x_train, y_train):
#     predicted = model(x)
#     print(x, np.array(np.round(predicted), dtype=int), y)