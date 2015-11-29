from __future__ import division
import numpy as np


class BaseLossFunction(object):

    def __call__(self, real, predicted, *args, **kwargs):
        raise NotImplementedError()

    def gradient(self, real, predicted, *args, **kwargs):
        raise NotImplementedError()


class LogLoss(BaseLossFunction):

    def __call__(self, real, predicted):
        pass

    def gradient(self, real, predicted):
        pass


class L2Loss(BaseLossFunction):

    def __call__(self, real, predicted):
        N = len(real)
        return 1.0 / N * np.sum((real - predicted)**2)

    def gradient(self, real, predicted):
        return predicted - real
