from __future__ import division
import autograd.numpy as np

from base import AutogradLoss


class MSE(AutogradLoss):

    def __call__(self, real, predicted):
        return np.mean((real - predicted)**2)
