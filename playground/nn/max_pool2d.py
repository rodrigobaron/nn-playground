import numpy as np
from .conv import Conv

class MaxPool2d(Conv):
    """TODO: MaxPool2d layer doc"""
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

        max_idx = np.argmax(x_col, axis=0)
        out = x_col[max_idx, range(max_idx.size)]

        self.x_col = x_col
        self.max_idx = max_idx

        out = out.reshape(h_out, w_out, n, d)
        return out.transpose(2, 3, 0, 1)

    def backward(self, grad):
        n, d, w, h = self.inputs.shape

        out_col = np.zeros_like(self.x_col)
        grad_col = grad.transpose(2, 3, 0, 1).ravel()

        out_col[self.max_idx, range(grad_col.size)] = grad_col

        out = self._col2im_indices(out_col, (n * d, 1, h, w))
        return out.reshape(self.inputs.shape)
