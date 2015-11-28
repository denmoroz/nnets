from __future__ import division
import numpy as np

from activations import SigmoidActivation
from initializations import uniform_init_weights, uniform_init_biases


class DenseLayer(object):

    def __init__(self, in_size, out_size,
                 activation=SigmoidActivation(),
                 init_weights=uniform_init_weights,
                 init_biases=uniform_init_biases):
        self.in_size = in_size
        self.out_size = out_size

        self._activation = activation

        # Transposed weights
        self._weights = init_weights(self.out_size, self.in_size)
        self._biases = init_biases(self.out_size)

        # Cache for learning algorithm
        # TODO: should be removed from layer class?
        self._weighted_inputs = None

    def weighted_inputs(self, prev_activation):
        return np.dot(self._weights, prev_activation) + self._biases

    def propagate_forward(self, prev_activation):
        self._weighted_inputs = self.weighted_inputs(prev_activation)
        return self._activation(self._weighted_inputs)

    def propagate_backward(self, next_weighted_delta):
        current_delta = next_weighted_delta * self._activation.derivative(self._weighted_inputs)
        weighted_delta = np.dot(self._weights.transpose(), current_delta)

        return current_delta, weighted_delta
