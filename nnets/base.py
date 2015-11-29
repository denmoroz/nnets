from __future__ import division
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