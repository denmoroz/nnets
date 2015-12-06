from __future__ import division
import numpy as np

from base import BaseLayer
from activation import SigmoidActivation
from initialization import uniform_init_weights, uniform_init_biases


class DenseLayer(BaseLayer):

    def __init__(self, in_size, out_size,
                 activation=SigmoidActivation(),
                 weights_regularization=None,
                 init_weights=uniform_init_weights,
                 init_biases=uniform_init_biases,
                 name=None):
        super(DenseLayer, self).__init__(name)

        self.in_size = in_size
        self.out_size = out_size

        self._activation = activation
        self._regularization = weights_regularization

        # Transposed weights
        self._weights = init_weights(self.out_size, self.in_size)
        self._biases = init_biases(self.out_size)

    @property
    def regularized(self):
        return self._regularization is not None

    @property
    def weights_regularization(self):
        if self.regularized:
            return self._regularization
        else:
            raise ValueError("Regularization is not set for layer: {}".format(self.name))

    def get_weighted_inputs(self, prev_activation):
        return np.dot(self._weights, prev_activation) + self._biases

    def propagate_forward(self, prev_activation):
        weighted_inputs = self.get_weighted_inputs(prev_activation)
        return weighted_inputs, self._activation(weighted_inputs)

    def propagate_backward(self, next_weighted_delta, current_weighted_inputs):
        current_delta = next_weighted_delta * self._activation.derivative(current_weighted_inputs)
        weighted_delta = np.dot(self._weights.transpose(), current_delta)

        return current_delta, weighted_delta
