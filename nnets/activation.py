from __future__ import division

import autograd.numpy as np

from base import AutogradActivation


class IdentityActivation(AutogradActivation):

    def __call__(self, z):
        return z


class SigmoidActivation(AutogradActivation):

    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-z))
