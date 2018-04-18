import numpy as np
from .layer import Layer
from .linear import Linear

class RNN(Layer):
    """TODO: RNN layer doc"""
    def __init__(self, timesteps, input_size, hidden_size, output_size, activation=None):
        super().__init__()
        self.params['w'] = np.random.randn(input_size + hidden_size, output_size)
        self.params['b'] = np.random.randn(output_size)

        self.params['wh'] = np.random.randn(input_size + hidden_size, hidden_size)
        self.params['bh'] = np.random.randn(hidden_size)
 
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, inputs):
        self.inputs = []
        self.hidden_state = np.zeros((inputs.shape[0], self.timesteps, self.hidden_size))
        
        new_shape = list(inputs.shape)
        new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
        new_shape = tuple(new_shape)

        step_inputs = np.copy(inputs)
        step_inputs = np.reshape(step_inputs, new_shape)

        outputs = []        
        for step in range(self.timesteps):
            combined = np.concatenate((step_inputs[step], self.hidden_state[:,step,:]), axis=1)
            self.inputs.append(combined)
            self.hidden_state[:,step,:] = combined @ self.params['wh'] + self.params['bh']
            if self.activation is not None:
                self.hidden_state[:,step,:] = self.activation(self.hidden_state[:,step,:])
            output = combined @ self.params['w'] + self.params['b']
            outputs.append(output)

        outputs = np.array(outputs)
        new_shape = list(outputs.shape)
        new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
        new_shape = tuple(new_shape)

        outputs = np.reshape(outputs, new_shape)
        return outputs, self.hidden_state

    def backward(self, grad):
        combined = self.inputs[-1]

        hidden = self.hidden_state[:, -1, :]
        if self.activation is not None:
            hidden = self.activation.backward(hidden)
        
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = combined.T @ grad

        self.grads['bh'] = np.sum(hidden, axis=0)
        self.grads['wh'] = combined.T @ hidden

        for step in range(self.timesteps -1):
            combined = self.inputs[step]
            hidden = self.hidden_state[:, step, :]
            self.grads['wh'] += combined.T @ hidden
        
        out = grad @ self.params['w'].T @ self.params['wh']

        return out  
