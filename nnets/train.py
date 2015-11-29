from __future__ import division

import random
import numpy as np

from base import LoggingMixin
from utils import mini_batch_iterator


class VanillaSGD(LoggingMixin, object):

    def __init__(self, model, loss, learning_rate,
                 epochs, batch_size, test_every):
        self.model = model

        self.loss = loss
        self.learning_rate = learning_rate

        self.epochs = epochs
        self.batch_size = batch_size
        self.test_every = test_every

    def _get_loss_derivatives(self, x, y):
        return self.model.get_loss_derivatives(self.loss, x, y)

    def _averaged_gradients(self, mini_batch):
        batch_size = len(mini_batch)
        avg_grad_w, avg_grad_b = {}, {}

        # Sum gradients for provided mini_batch
        for x, y in mini_batch:
            weights_grad, biases_grad = self._get_loss_derivatives(x, y)

            for layer in self.model.layers:
                if layer.name not in avg_grad_w:
                    avg_grad_w[layer.name] = weights_grad[layer.name]
                    avg_grad_b[layer.name] = biases_grad[layer.name]
                else:
                    avg_grad_w[layer.name] += weights_grad[layer.name]
                    avg_grad_b[layer.name] += biases_grad[layer.name]

        for layer in self.model.layers:
            avg_grad_w[layer.name] /= batch_size
            avg_grad_b[layer.name] /= batch_size

        return avg_grad_w, avg_grad_b

    def update_mini_batch(self, mini_batch):
        avg_grad_w, avg_grad_b = self._averaged_gradients(mini_batch)

        for layer in self.model.layers:
            # X = X - learning_rate * average_gradient
            layer._weights -= self.learning_rate * avg_grad_w[layer.name]
            layer._biases -= self.learning_rate * avg_grad_b[layer.name]

    def evaluate(self, data):
        n = len(data)
        real, predicted = np.zeros(n), np.zeros(n)

        for idx, (x, y) in enumerate(data):
            predicted[idx] = self.model.predict(x)
            real[idx] = y

        return self.loss(real, predicted)

    def fit(self, train_data, test_data=None):
        if test_data is not None:
            random_loss = self.evaluate(test_data)
            self.log('Random model loss: {}'.format(random_loss))

        for epoch_num in xrange(self.epochs):
            random.shuffle(train_data)

            batch_iter = mini_batch_iterator(train_data, self.batch_size)
            for batch_num, next_batch in enumerate(batch_iter):
                self.update_mini_batch(next_batch)

                if test_data is not None and batch_num % self.test_every == 0:
                    current_loss = self.evaluate(test_data)
                    self.log('Epoch num: {}, batch num: {}, loss:{}'.format(epoch_num, batch_num, current_loss))
