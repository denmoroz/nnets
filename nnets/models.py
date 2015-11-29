from __future__ import division
from itertools import izip

import numpy as np

from storage import DynamicStorage


class MLPModel(object):

    def __init__(self, layers):
        self.layers = layers
        self._storage = DynamicStorage()

    def get_loss_derivatives(self, loss, x, y):
        # Forward propagation
        activation = x
        for layer in self.layers:
            weighted_input, activation = layer.propagate_forward(activation)

            self._storage[layer.name]['input'] = weighted_input
            self._storage[layer.name]['activation'] = activation

        # Backward propagation
        last_layer = self.layers[-1]
        output_activation = self._storage[last_layer.name]['activation']
        weighted_delta = loss.gradient(y, output_activation)

        for layer in reversed(self.layers):
            layer_input = self._storage[layer.name]['input']

            # Delta's will be stored for parameters update, weighted delta will be backpropagated
            delta, weighted_delta = layer.propagate_backward(weighted_delta, layer_input)
            self._storage[layer.name]['delta'] = delta

        layer_names = [layer.name for layer in self.layers]
        activations = [x] + self._storage.for_keys(layer_names[:-1], 'activation')
        deltas = self._storage.for_keys(layer_names, 'delta')

        # Calculate cost function derivatives for each w_{i,j}^{l}, b_{j}^{l}, where l - layer number
        nabla_w, nabla_b = [], []
        for delta, activation in izip(deltas, activations):
            delta_w = np.outer(delta, activation)
            delta_b = delta

            nabla_w.append(delta_w)
            nabla_b.append(delta_b)

        return nabla_w, nabla_b

    def predict(self, x):
        # Activations of the first layer = x
        activation = x

        for layer in self.layers:
            # Drop weighted inputs of layer - its only necessary for learning phase
            _, activation = layer.propagate_forward(activation)

        return activation

    def __getitem__(self, layer_index):
        return self.layers[layer_index]
