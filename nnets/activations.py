from __future__ import division
import numpy as np


class BaseActivation(object):
    """
        Base class for all activations.
        All subclasses should implement defined methods in element-wise manner.
    """
    def __call__(self, z):
        raise NotImplementedError()

    def derivative(self, z):
        raise NotImplementedError()


class IdentityActivation(BaseActivation):

    def __call__(self, z):
        return z

    def derivative(self, z):
        return 1.0


class SigmoidActivation(BaseActivation):

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def __call__(self, z):
        return self.sigmoid(z)

    def derivative(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))
