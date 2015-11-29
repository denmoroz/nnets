from __future__ import division

import random
import numpy as np

from base import LoggingMixin
from utils import mini_batch_iterator


class VanillaSGD(LoggingMixin, object):

    def __init__(self, loss, epochs, batch_size, learning_rate, test_every):
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_every = test_every
        self.learning_rate = learning_rate

    def update_mini_batch(self, model, mini_batch):
        n = len(mini_batch)

        sum_delta_w = [np.zeros(layer._weights.shape) for layer in model.layers]
        sum_delta_b = [np.zeros(layer._biases.shape) for layer in model.layers]

        # Sum of gradients for provided mini_batch
        # TODO: may be done in parallel with multiprocessing pool
        for x, y in mini_batch:
            delta_w, delta_b = model.get_loss_derivatives(self.loss, x, y)

            for l in xrange(len(model.layers)):
                sum_delta_w[l] += delta_w[l]
                sum_delta_b[l] += delta_b[l]

        for layer_idx, layer in enumerate(model.layers):
            # X = X - learning_rate * average_gradient
            layer._weights -= self.learning_rate * sum_delta_w[layer_idx] / n
            layer._biases -= self.learning_rate * sum_delta_b[layer_idx] / n

    def evaluate(self, model, data):
        n = len(data)
        real, predicted = np.zeros(n), np.zeros(n)

        for idx, (x, y) in enumerate(data):
            predicted[idx] = model.predict(x)
            real[idx] = y

        return self.loss(real, predicted)

    def fit(self, model, train_data, test_data=None):
        if test_data is not None:
            random_loss = self.evaluate(model, test_data)
            self.log('Random model loss: {}'.format(random_loss))

        for epoch_num in xrange(self.epochs):
            random.shuffle(train_data)

            batch_iter = mini_batch_iterator(train_data, self.batch_size)
            for batch_num, next_batch in enumerate(batch_iter):
                self.update_mini_batch(model, next_batch)

                if test_data is not None and batch_num % self.test_every == 0:
                    current_loss = self.evaluate(model, test_data)
                    self.log('Epoch num: {}, batch num: {}, loss:{}'.format(epoch_num, batch_num, current_loss))
