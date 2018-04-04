import numpy as np
from .conv import Conv

class AvgPool2d(Conv):
    """TODO: AvgPool2d layer doc"""
    def __init__(self, kernel_size, stride=1):
        super().__init__(kernel_size, stride, 0)

    def forward(self, inputs):
        self.inputs = inputs
        n, d, h, w = inputs.shape
        h_out = (h - self.kernel_size_h) / self.stride_h + 1
        w_out = (w - self.kernel_size_w) / self.stride_w + 1

        h_out, w_out = int(h_out), int(w_out)

        inputs_reshaped = inputs.reshape(n * d, 1, h, w)
        x_col = self._im2col_indices(inputs_reshaped)

        out = np.mean(x_col, axis=0)

        self.x_col = x_col

        out = out.reshape(h_out, w_out, n, d)
        return out.transpose(2, 3, 0, 1)

    def backward(self, grad):
        n, d, w, h = self.inputs.shape

        out_col = np.zeros_like(self.x_col)
        grad_col = grad.transpose(2, 3, 0, 1).ravel()

        out_col[:, range(grad_col.size)] = 1. / out_col.shape[0] * grad_col

        out = self._col2im_indices(out_col, (n * d, 1, h, w))
        return out.reshape(self.inputs.shape)
