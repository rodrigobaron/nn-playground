import os.path as P
import struct

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

def load_mnist(train=False, one_hot=True):
    base = P.join(P.dirname(__file__), 'mnist_data')
    if train:
        img_p = P.join(base, 'train-images.idx3-ubyte')
        label_p = P.join(base, 'train-labels.idx1-ubyte')
    else:
        img_p = P.join(base, 't10k-images.idx3-ubyte')
        label_p = P.join(base, 't10k-labels.idx1-ubyte')

    with open(label_p, 'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.int8)

    with open(img_p, 'rb') as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), rows, cols)

    if one_hot:
        _one_hot = np.zeros((labels.shape[0], 10))
        _one_hot[np.arange(labels.shape[0]), labels] = 1
        labels = _one_hot

    return images, labels
    # images = []
    # labels = []
    #
    # for i in range(len(lbl)):
    #     images.append(img[i])
    #     labels.append(lbl[i])
    #
    # return np.array(images), np.array(labels)




    # get_img = lambda idx: (lbl[idx], img[idx])
    #
    # for i in range(len(lbl)):
    #     yield get_img(i)
#
# def show(image):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
#     imgplot.set_interpolation('nearest')
#     ax.xaxis.set_ticks_position('top')
#     ax.yaxis.set_ticks_position('left')
#     plt.show()