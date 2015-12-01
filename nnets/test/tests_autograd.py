from nose.tools import ok_
from nose.tools import nottest

import numpy as np
from ..loss import MSE, MAE, BinaryCrossEntropy
from ..activation import IdentityActivation, SigmoidActivation, TanhActivation, HardTanhActivation, RectifierActivation


def test_mse():
    mse = MSE()

    real = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    predicted = np.array([0.99, 1.87, 1.1, 2.7, 0.95])

    # Real gradient for MSE, for provided real an predicted arrays
    def real_gradient(real, predicted):
        return 2.0 * (predicted - real)

    ok_(np.allclose(mse.gradient(real, predicted), real_gradient(real, predicted)))


def test_mae():
    mae = MAE()

    real = np.array([1.0, 2.0, 1.0, 3.0])
    predicted = np.array([0.99, 1.87, 1.1, 2.7])

    def real_gradient(real, predicted):
        return np.sign(predicted - real)

    ok_(np.allclose(mae.gradient(real, predicted), real_gradient(real, predicted)))


def test_binary_cross_entropy():
    log_loss = BinaryCrossEntropy()

    real = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    predicted = np.array([0.05, 0.17, 0.65, 0.99, 0.87])

    def real_gradient(real, predicted):
        return -(real/predicted - (1.0-real)/(1.0-predicted))

    ok_(np.allclose(log_loss.gradient(real, predicted), real_gradient(real, predicted)))


def test_identity_activation():
    z = 100.0

    identity = IdentityActivation()
    ok_(np.isclose(identity.derivative(z), 1.0))


def test_sigmoid_activation():
    z = 100.0

    sigmoid = SigmoidActivation()

    def real_derivative(z):
        return sigmoid(z)*(1.0 - sigmoid(z))

    ok_(np.isclose(sigmoid.derivative(z), real_derivative(z)))


def test_tanh_activation():
    z = 100.0

    tanh = TanhActivation()

    def real_derivative(z):
        return 1.0 - tanh(z)**2

    ok_(np.isclose(tanh.derivative(z), real_derivative(z)))


def test_hardtanh_activation():
    z = 100.0

    hard_tanh = HardTanhActivation()

    def real_derivative(z):
        if -1.0 < z < 1.0:
            return 1.0
        else:
            return 0.0

    ok_(np.isclose(hard_tanh.derivative(z), real_derivative(z)))


def test_rectifier_activation():
    z = 100.0

    relu = RectifierActivation()

    def real_derivative(z):
        if z > 0.0:
            return 1.0
        else:
            return 0.0

    ok_(np.isclose(relu.derivative(z), real_derivative(z)))
