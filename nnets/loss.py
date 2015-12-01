from __future__ import division
import autograd.numpy as np

from base import AutogradLoss


class MSE(AutogradLoss):

    def values(self, real, predicted):
        return (real - predicted)**2


class MAE(AutogradLoss):

    def values(self, real, predicted):
        return np.abs(real - predicted)


class BinaryCrossEntropy(AutogradLoss):

    def __init__(self, eps=1e-15):
        super(BinaryCrossEntropy, self).__init__()
        self.eps = eps

    def values(self, real, predicted):
        return -real*np.log(predicted)-(1.0-real)*np.log(1.0-predicted)

    def preprocess(self, real, predicted):
        # Since log loss is undefined in p = 0.0 and p = 1.0
        predicted[predicted < self.eps] = self.eps
        predicted[predicted > 1.0-self.eps] = 1.0-self.eps

        return real, predicted


class CategoricalCrossEntropy(AutogradLoss):

    def values(self, real, predicted):
        return -np.dot(real, np.log(predicted))
