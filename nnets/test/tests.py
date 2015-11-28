from nose.tools import ok_

import numpy as np

from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split

from ..models import MLPModel
from ..layers import DenseLayer
from ..costs import L2Loss
from ..train import VanillaSGD

from ..activations import SigmoidActivation, IdentityActivation
from ..initializations import const_init_weights, const_init_biases


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

    loss = L2Loss()

    x = np.array([1.0, 1.0])
    y = np.array([1.0])

    nabla_w, nabla_b = mlp.get_cost_derivatives(loss, x, y)
    ok_(len(nabla_w) == len(nabla_b))

    for idx, layer in enumerate(mlp.layers):
        ok_(layer._weights.shape == nabla_w[idx].shape)
        ok_(layer._biases.shape == nabla_b[idx].shape)


def test_vanilla_sgd():
    mlp = MLPModel(
        layers=[
            DenseLayer(in_size=1, out_size=1, activation=IdentityActivation()),
        ]
     )

    sgd = VanillaSGD(loss=L2Loss(), epochs=5, batch_size=128, learning_rate=0.2, test_every=2)

    X, y, real_coef = make_regression(n_samples=1000, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data, test_data = zip(X_train, y_train), zip(X_test, y_test)

    sgd.fit(mlp, train_data, test_data)

    estimated_coef = mlp[0]._weights[0][0]
    ok_(abs(real_coef-estimated_coef) <= 2.5)
