# import os.path as P
# import struct
#
#
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np

from .downloadable import Downloadable
import gzip

class MNISTData(Downloadable):
    def __init__(self):
        super().__init__('http://yann.lecun.com/exdb/mnist/')
        self.train_images = 'train-images-idx3-ubyte.gz'
        self.train_labels = 'train-labels-idx1-ubyte.gz'
        self.test_images = 't10k-images-idx3-ubyte.gz'
        self.test_labels = 't10k-labels-idx1-ubyte.gz'

        self.all_data = {
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'test_images': self.test_images,
            'test_labels': self.test_labels
        }

    def extract_images(self, local_file, silent=False):
        if not silent:
            print('Extracting', local_file)
        with gzip.open(local_file) as bytestream:
            _ = self.read_numpy32(bytestream)
            # if magic != 2051:
            #     raise ValueError(
            #         'Invalid magic number %d in MNIST image file: %s' %
            #         (magic, filename))
            num_images = self.read_numpy32(bytestream)
            rows = self.read_numpy32(bytestream)
            cols = self.read_numpy32(bytestream)
            s = rows * cols * num_images
            buf = bytestream.read(s[0])
            data = np.frombuffer(buf, dtype=np.uint8)

            data = data.reshape(num_images[0], rows[0], cols[0], 1)

            return data

    def extract_labels(self, local_file, silent=False):
        if not silent:
            print('Extracting', local_file)
        with gzip.open(local_file) as bytestream:
            _ = self.read_numpy32(bytestream)
            # if magic != 2049:
            #     raise ValueError(
            #         'Invalid magic number %d in MNIST label file: %s' %
            #         (magic, local_file))
            num_items = self.read_numpy32(bytestream)
            buf = bytestream.read(num_items[0])
            labels = np.frombuffer(buf, dtype=np.uint8)
            # if one_hot:
            #     return dense_to_one_hot(labels)
            return labels

    def load(self, target_dir, silent=False, one_hot=False):
        data_sets = {}
        for key in self.all_data.keys():
            local_file = self.try_download(self.all_data[key], target_dir)
            if 'labels' in key:
                data = self.extract_labels(local_file, silent=silent)
                if one_hot:
                    _one_hot = np.zeros((data.shape[0], 10))
                    _one_hot[np.arange(data.shape[0]), data] = 1
                    data = _one_hot
            else:
                data = self.extract_images(local_file, silent=silent)

            data_sets.update({key: data})

        return (data_sets['train_images'], data_sets['train_labels']), \
               (data_sets['test_images'], data_sets['test_labels'])




#
# def load_mnist(train=False, one_hot=True):
#     base = P.join(P.dirname(__file__), 'mnist_data')
#     if train:
#         img_p = P.join(base, 'train-images.idx3-ubyte')
#         label_p = P.join(base, 'train-labels.idx1-ubyte')
#     else:
#         img_p = P.join(base, 't10k-images.idx3-ubyte')
#         label_p = P.join(base, 't10k-labels.idx1-ubyte')
#
#     with open(label_p, 'rb') as f:
#         _, n = struct.unpack(">II", f.read(8))
#         labels = np.fromfile(f, dtype=np.int8)
#
#     with open(img_p, 'rb') as f:
#         _, n, rows, cols = struct.unpack(">IIII", f.read(16))
#         images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), rows, cols)
#
#     if one_hot:
#         _one_hot = np.zeros((labels.shape[0], 10))
#         _one_hot[np.arange(labels.shape[0]), labels] = 1
#         labels = _one_hot
#
#     return images, labels
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