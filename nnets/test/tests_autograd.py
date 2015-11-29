from nose.tools import ok_

import numpy as np
from ..loss import MSE
from ..activation import IdentityActivation, SigmoidActivation


def test_mse():
    mse = MSE()

    real = np.array([1.0, 2.0, 1.0, 3.0])
    predicted = np.array([0.99, 1.87, 1.1, 2.7])

    # Real gradient for MSE, for provided real an predicted arrays
    def real_gradient(real, predicted):
        return  0.5 * (predicted - real)

    ok_(np.allclose(mse.gradient(real, predicted), real_gradient(real, predicted)))


def test_identity_activation():
    z = 100.0

    identity = IdentityActivation()
    ok_(np.isclose(identity.derivative(z), 1.0))


def test_sigmoid_activation():
    z = 100.0

    sigmoid = SigmoidActivation()

    def real_derivative(z):
        return  sigmoid(z)*(1.0 - sigmoid(z))

    ok_(np.isclose(sigmoid.derivative(z), real_derivative(z)))
