from __future__ import division
import autograd.numpy as np

from base import AutogradLoss


class MSE(AutogradLoss):

    def value(self, real, predicted):
        return np.mean((real - predicted)**2)


class BinaryCrossEntropy(AutogradLoss):

    def __init__(self, eps=1e-15):
        super(BinaryCrossEntropy, self).__init__()
        self.eps = eps

    def value(self, real, predicted):
        return np.mean(-real*np.log(predicted)-(1.0-real)*np.log(predicted))

    def preprocess(self, real, predicted):
        # Since log loss is undefined in p = 0.0 and p = 1.0
        predicted[predicted < self.eps] = self.eps
        predicted[predicted > 1.0-self.eps] = 1.0-self.eps

        return real, predicted
