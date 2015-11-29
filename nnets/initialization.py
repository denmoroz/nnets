import numpy as np


def uniform_init_weights(n, m, eps=0.01):
    return np.random.uniform(-eps, eps, size=(n, m))


def uniform_init_biases(n, eps=0.01):
    return np.random.uniform(-eps, eps, size=n)


def const_init_weights(n, m, C=1.0):
    """ For testing purposes only! """
    return C * np.ones(shape=(n, m))

def const_init_biases(n, C=1.0):
    """ For testing purposes only! """
    return C * np.ones(shape=n)
