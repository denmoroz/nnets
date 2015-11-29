from __future__ import division
from itertools import izip

import numpy as np


class MLPModel(object):

    def __init__(self, layers):
        self.layers = layers

    def get_loss_derivatives(self, loss, x, y):
        activations, deltas = [x], []

        # Forward propagation
        # Initial activation is input
        activation = activations[0]
        for layer in self.layers:
            activation = layer.propagate_forward(activation)
            activations.append(activation)

        # Backward propagation
        weighted_delta = loss.gradient(y, activations[-1])
        for layer in reversed(self.layers):
            # Delta's will be stored for weights update, weighted delta will be backpropagated
            delta, weighted_delta = layer.propagate_backward(weighted_delta)
            deltas.append(delta)

        # Last activation is only needed to initialize backpropagation
        activations = activations[:-1]
        # Deltas collected in reverse order
        deltas = reversed(deltas)

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
            activation = layer.propagate_forward(activation)

        return activation

    def __getitem__(self, layer_index):
        return self.layers[layer_index]
