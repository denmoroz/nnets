from __future__ import division
import logging

from autograd import grad, elementwise_grad


class BaseActivation(object):
    """
        Base class for all activations.
    """
    def __call__(self, z):
        raise NotImplementedError()

    def derivative(self, z):
        raise NotImplementedError()


class AutogradActivation(BaseActivation):
    """
        Activation function with automatic differentiation.
        All math logic inside __call__ method should be
        implemented with autograd.numpy wrapper.
    """
    def __init__(self):
        self._derivative = elementwise_grad(self.__call__)

    def derivative(self, z):
        return self._derivative(z)


class BaseLoss(object):
    """
        Base class for all loss functions.
    """
    def __call__(self, real, predicted):
        raise NotImplementedError()

    def gradient(self, real, predicted):
        raise NotImplementedError()


class AutogradLoss(BaseLoss):
    """
        Loss function with automatic differentiation.
        All math logic inside __call__ method should be
        implemented with autograd.numpy wrapper.
    """
    def __init__(self):
        # Gradient by "predicted" variable
        self._grad = grad(self.__call__, argnum=1)

    def gradient(self, real, predicted):
        return self._grad(real, predicted)


class BaseLayer(object):
    __layers_created = 0

    @classmethod
    def _get_next_name(cls):
        next_name = '{}_{}'.format(cls.__name__, cls.__layers_created)
        cls.__layers_created += 1
        return next_name

    def __init__(self, name=None):
        if name is None:
            self.name = self._get_next_name()
        else:
            self.name = name


class LoggingMixin(object):

    def log(self, msg, level=logging.INFO):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)

        self._logger.log(level, msg)
