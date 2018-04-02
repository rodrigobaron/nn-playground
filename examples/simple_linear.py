import sys
sys.path.append(sys.path[0] + "/..")

import numpy as np

from playground.nn import Model, CrossEntropyLoss
from playground.nn import Linear
from playground.utils import BatchIterator
from playground.optim import SGD
from playground.reg import L2Regularization
from playground.data import MNISTData

def pre_process_images(x_data):
    x_data = x_data.astype(np.float32)
    mean = np.mean(x_data)
    x_data = x_data - mean
    x_data[:] /= 255.0
    return x_data

img_size = 28
num_classes = 10
img_size_flat = img_size * img_size


data_loader = MNISTData()
(x_train, y_train), (x_test, y_test) = data_loader.load('data/MNIST')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

x_train, x_test = pre_process_images(x_train), pre_process_images(x_test)

val_size = int(len(x_train) * 0.2)

x_val, y_val = x_train[val_size:], y_train[val_size:]
x_train, y_train = x_train[:val_size], y_train[:val_size]

print('---' * 10)
print('shape x_train', x_train.shape)
print('shape y_train', y_train.shape)
print('---' * 10)
print('shape x_train', x_val.shape)
print('shape y_train', y_val.shape)
print('---' * 10)
print('shape x_train', x_test.shape)
print('shape y_train', y_test.shape)
print('---' * 10)

model = Model([
    Linear(input_size=img_size_flat, output_size=num_classes)
])

iterator = BatchIterator(batch_size=100, shuffle=True)
regularization = L2Regularization(model)
loss = CrossEntropyLoss()

optimizer = SGD(lr=0.01)
epochs = 100

for i, epoch in enumerate(range(epochs)):
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    k = 0
    for x_batch, y_true_batch in iterator(x_train, y_train):
        predicted = model(x_batch)
        train_epoch_loss += loss(predicted, y_true_batch)
        grad = loss.grad(predicted, y_true_batch)
        model.backward(grad)
        optimizer.step(model)

    for x_batch, y_true_batch in iterator(x_val, y_val):
        predicted = model(x_batch)
        val_epoch_loss += loss(predicted, y_true_batch)

    if i % 10 == 9:
        print("epoch: %d, loss: %.5f, val_loss: %0.5f" % (epoch+1, train_epoch_loss, val_epoch_loss))

corrects = 0
count = 0
for x, y in zip(x_test, y_test):
    predicted = np.argmax(model(x))
    if predicted == y:
        status = "OK"
        corrects += 1
    else:
        status = "FAIL"
    count += 1
print("Evaluation on testset: %d/%d (%.2f %%)" % (corrects, count, corrects / count * 100))