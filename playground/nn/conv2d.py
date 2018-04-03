import numpy as np
from .layer import Layer

class Conv2d(Layer):
    """TODO: Conv2d layer doc"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.params['w'] = np.random.randn(in_channels, out_channels)
        self.params['b'] = np.random.randn(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, inputs):
        # self.inputs = inputs
        n_filters, d_filter, h_filter, w_filter = self.params['w'].shape
        n_x, d_x, h_x, w_x = inputs.shape
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1

        h_out, w_out = int(h_out), int(w_out)
        self.inputs = self.im2col_indices(inputs, h_filter, w_filter, padding=self.padding, stride=self.stride)
        self.params['w'] = self.params['w'].reshape(n_filters, -1)

        return self.inputs @ self.params['w'] + self.params['b']
        # return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        n_filter, d_filter, h_filter, w_filter = W.shape

        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['b'] = self.grads['b'].reshape(n_filter, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        self.grads['w'] = grad_reshaped @ self.inputs.T
        self.grads['w'] = self.grads['w'].reshape(self.params['w'].shape)

        W_reshape = self.params['w'].reshape(n_filter, -1)
        dX_col = W_reshape.T @ grad_reshaped
        return self.col2im_indices(dX_col, self.inputs.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)

        # self.grads['w'] = self.inputs.T @ grad
        # return grad @ self.params['w'].T


    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_height) % stride == 0
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k.astype(int), i.astype(int), j.astype(int))

    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding, stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]
