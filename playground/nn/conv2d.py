import numpy as np
from .conv import Conv

class Conv2d(Conv):
    """TODO: Conv2d layer doc"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(kernel_size, stride, padding)

        self.params['w'] = np.random.randn(out_channels, in_channels, self.kernel_size_h, self.kernel_size_w)
        self.params['b'] = np.random.randn(out_channels, 1)

        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, inputs):
        self.inputs = inputs
        n_x, d_x, h_x, w_x = inputs.shape
        h_out = (h_x - self.kernel_size_h + 2 * self.padding_h) / self.stride_h + 1
        w_out = (w_x - self.kernel_size_w + 2 * self.padding_w) / self.stride_w + 1

        h_out, w_out = int(h_out), int(w_out)

        x_col = self._im2col_indices(inputs)
        w_col = self.params['w'].reshape(self.out_channels, -1)

        out = w_col @ x_col + self.params['b']
        out = out.reshape(self.out_channels, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        self.x_col = x_col

        return out

    def backward(self, grad):
        db = np.sum(grad, axis=(0, 2, 3))
        self.grads['b'] = db.reshape(self.out_channels, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.grads['w'] = (grad_reshaped @ self.x_col.T).reshape(self.params['w'].shape)

        w_reshape = self.params['w'].reshape(self.out_channels, -1)
        dX_col = w_reshape.T @ grad_reshaped
        return self._col2im_indices(dX_col, self.inputs.shape)
