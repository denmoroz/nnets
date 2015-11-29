from __future__ import division
from itertools import izip

import numpy as np


class MLPModel(object):

    def __init__(self, layers):
        self.layers = layers
        self._mapping = {layer.name: idx for idx, layer in enumerate(self.layers)}

    def get_loss_derivatives(self, loss, x, y):
        inputs, activations, deltas = {}, {}, {}

        # Forward propagation
        activation = x
        for layer in self.layers:
            weighted_input, activation = layer.propagate_forward(activation)

            inputs[layer.name] = weighted_input
            activations[layer.name] = activation

        # Backward propagation
        last_layer = self.layers[-1]
        output_activation = activations[last_layer.name]
        weighted_delta = loss.gradient(y, output_activation)

        for layer in reversed(self.layers):
            layer_input = inputs[layer.name]

            # Delta's will be stored for parameters update, weighted delta will be backpropagated
            delta, weighted_delta = layer.propagate_backward(weighted_delta, layer_input)
            deltas[layer.name] = delta

        layer_names = [layer.name for layer in self.layers]
        activations = [x] + [activations[name] for name in layer_names[:-1]]
        deltas = [deltas[name] for name in layer_names]

        # Calculate cost function derivatives for each layer's weights and biases
        weights_grad, biases_grad = {}, {}
        for layer_name, delta, activation in izip(layer_names, deltas, activations):
            weights_grad[layer_name] = np.outer(delta, activation)
            biases_grad[layer_name] = delta

        return weights_grad, biases_grad

    def predict(self, x):
        # Activations of the first layer = x
        activation = x

        for layer in self.layers:
            # Drop weighted inputs of layer - its only necessary for learning phase
            _, activation = layer.propagate_forward(activation)

        return activation

    def layer_by_index(self, layer_idx):
        return self.layers[layer_idx]

    def layer_by_name(self, layer_name):
        return self.layer_by_index(self._mapping[layer_name])
