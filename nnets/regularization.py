from __future__ import division

import autograd.numpy as np
from autograd import elementwise_grad


class BaseWeightsRegularization(object):

    def __init__(self, C):
        self.C = C

    def __call__(self, weights):
        return np.sum(self.values(weights))

    def values(self, weights):
        raise NotImplementedError()

    def gradient(self, weights):
        raise NotImplementedError()


class AutogradWeightsRegularization(BaseWeightsRegularization):

    def __init__(self, C):
        super(AutogradWeightsRegularization, self).__init__(C)
        self._grad = elementwise_grad(self.values)

    def gradient(self, weights):
        return self._grad(weights)


class L1Regularization(AutogradWeightsRegularization):

    def values(self, weights):
        return self.C * np.abs(weights)


class L2Regularization(AutogradWeightsRegularization):

    def values(self, weights):
        return self.C * weights ** 2


class ElasticNetRegularization(AutogradWeightsRegularization):

    def values(self, weights):
        return 0.5 * self.C * weights ** 2 + (1.0 - self.C) * np.abs(weights)