from __future__ import division

import autograd.numpy as np

from base import BaseActivation, AutogradActivation


class IdentityActivation(BaseActivation):

    def __call__(self, z):
        return z

    def derivative(self, z):
        return 1.0


class SigmoidActivation(AutogradActivation):

    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-z))


class TanhActivation(AutogradActivation):

    def __call__(self, z):
        return np.tanh(z)


class HardTanhActivation(AutogradActivation):

    def __call__(self, z):
        result = z

        if z < -1.0:
            result = -1.0
        elif z > 1.0:
            result = 1.0

        return result


class RectifierActivation(AutogradActivation):

    def __call__(self, z):
        if z > 0.0:
            return z
        else:
            return 0.0
