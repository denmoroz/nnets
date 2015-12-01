from nose.tools import ok_
from nose.tools import nottest

import numpy as np

from sklearn.datasets import make_regression, make_blobs
from sklearn.cross_validation import train_test_split

from ..models import MLPModel
from ..layers import DenseLayer

from ..activation import SigmoidActivation, IdentityActivation
from ..loss import MSE, BinaryCrossEntropy
from ..initialization import const_init_weights, const_init_biases

from ..train import VanillaSGD


def test_forward_propagation():
    mlp = MLPModel(
        # 2-3-1 network
        layers=[
            DenseLayer(in_size=2, out_size=3,
                       activation=SigmoidActivation(),
                       init_weights=const_init_weights,
                       init_biases=const_init_biases),

            DenseLayer(in_size=3, out_size=1,
                       activation=SigmoidActivation(),
                       init_weights=const_init_weights,
                       init_biases=const_init_biases),
        ]
    )

    x = np.array([1.0, 1.0])
    output_activation = mlp.predict(x)

    ok_(np.isclose(output_activation, 0.9793206))


def test_one_pass():
    mlp = MLPModel(
        # 2-3-1 network
        layers=[
            DenseLayer(in_size=2, out_size=3,
                       activation=SigmoidActivation(),
                       init_weights=const_init_weights,
                       init_biases=const_init_biases),

            DenseLayer(in_size=3, out_size=1,
                       activation=SigmoidActivation(),
                       init_weights=const_init_weights,
                       init_biases=const_init_biases),
        ]
    )

    loss = MSE()

    x = np.array([1.0, 1.0])
    y = np.array([1.0])

    weights_grad, biases_grad = mlp.get_loss_derivatives(loss, x, y)

    for layer in mlp.layers:
        ok_(layer._weights.shape == weights_grad[layer.name].shape)
        ok_(layer._biases.shape == biases_grad[layer.name].shape)


def test_vanilla_sgd_regression():
    mlp = MLPModel(
        layers=[
            DenseLayer(in_size=3, out_size=1, activation=IdentityActivation()),
        ]
     )

    sgd = VanillaSGD(model=mlp, loss=MSE(), learning_rate=0.2,
                     epochs=5, batch_size=128, test_every=2)

    X, y, real_coef = make_regression(n_samples=1000, n_features=3,
                                      n_informative=3, noise=0,
                                      coef=True, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data, test_data = zip(X_train, y_train), zip(X_test, y_test)

    sgd.fit(train_data, test_data)
    estimated_coef = mlp.layer_by_index(0)._weights

    ok_(np.allclose(real_coef, estimated_coef))


def test_vanilla_sgd_binary_classification():
    # Single layer perceptron
    mlp = MLPModel(
        layers=[
            DenseLayer(in_size=3, out_size=1, activation=SigmoidActivation()),
        ]
     )

    sgd = VanillaSGD(model=mlp, loss=BinaryCrossEntropy(), learning_rate=0.2,
                     epochs=5, batch_size=128, test_every=2)

    X, y = make_blobs(n_samples=1000, centers=2, n_features=3, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data, test_data = zip(X_train, y_train), zip(X_test, y_test)

    sgd.fit(train_data, test_data)
    fit_loss = sgd.evaluate(test_data)

    ok_(fit_loss <= 0.05)
