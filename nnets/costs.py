from __future__ import division
import numpy as np


class BaseLossFunction(object):

    def evaluate(self, real, predicted):
        raise NotImplementedError()

    def derivative(self, real, predicted):
        raise NotImplementedError()


class LogLoss(BaseLossFunction):

    def evaluate(self, real, predicted):
        pass

    def derivative(self, real, predicted):
        pass


class L2Loss(BaseLossFunction):

    def evaluate(self, real, predicted):
        N = len(real)
        return 1.0 / N * np.sum((real - predicted)**2)

    def derivative(self, real, predicted):
        return predicted - real
