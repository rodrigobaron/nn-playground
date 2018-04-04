import numpy as np
from .layer import Layer

class Conv(Layer):
    """TODO: Conv layer doc"""
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()

        if type(kernel_size) is int:
            kernel_size_h = kernel_size
            kernel_size_w = kernel_size
        elif type(kernel_size) is tuple and len(kernel_size) == 2:
            kernel_size_h = kernel_size[0]
            kernel_size_w = kernel_size[1]
        else:
            raise Exception("`kernel_size` should be a int or a tuple of lenght 2")

        if type(stride) is int:
            stride_h = stride
            stride_w = stride
        elif type(stride) is tuple and len(stride) == 2:
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            raise Exception("`stride` should be a int or a tuple of lenght 2")

        if type(padding) is int:
            padding_h = padding
            padding_w = padding
        elif type(padding) is tuple and len(padding) == 2:
            padding_h = padding[0]
            padding_w = padding[1]
        else:
            raise Exception("`padding` should be a int or a tuple of lenght 2")

        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w

        self.stride_h = stride_h
        self.stride_w = stride_w

        self.padding_h = padding_h
        self.padding_w = padding_w

    def _get_im2col_indices(self, x_shape):
        N, C, H, W = x_shape
        assert (H + 2 * self.padding_h - self.kernel_size_h) % self.stride_h == 0
        assert (W + 2 * self.padding_w - self.kernel_size_w) % self.stride_w == 0

        out_height = int((H + 2 * self.padding_h - self.kernel_size_h) / self.stride_h + 1)
        out_width = int((W + 2 * self.padding_w - self.kernel_size_w) / self.stride_w + 1)

        i0 = np.repeat(np.arange(self.kernel_size_h), self.kernel_size_w)
        i0 = np.tile(i0, C)

        i1 = self.stride_h * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(self.kernel_size_w), self.kernel_size_h * C)
        j1 = self.stride_w * np.tile(np.arange(out_width), out_height)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), self.kernel_size_h * self.kernel_size_w).reshape(-1, 1)

        return (k.astype(int), i.astype(int), j.astype(int))

    def _im2col_indices(self, x):
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding_h, self.padding_w), (self.padding_h, self.padding_w)), mode='constant')

        k, i, j = self._get_im2col_indices(x.shape)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_size_h * self.kernel_size_w * C, -1)

        return cols

    def _col2im_indices(self, cols, x_shape):
        N, C, H, W = x_shape

        H_padded, W_padded = H + 2 * self.padding_h, W + 2 * self.padding_w
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

        k, i, j = self._get_im2col_indices(x_shape)

        cols_reshaped = cols.reshape(C * self.kernel_size_h * self.kernel_size_w, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

        if self.padding_h == 0 and self.padding_w == 0:
            return x_padded

        return x_padded[:, :, self.padding_h:-self.padding_h, self.padding_w:-self.padding_w]
