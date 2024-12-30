import numpy as np


def activation_func(x, func_name):
    return activation_funcs[func_name](x)

def activation_derivative_func(x, func_name):
    return activation_derivative_funcs[func_name](x)


activation_funcs = {
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'relu': lambda x: np.maximum(x, 0),
    'tanh': lambda x: np.tanh(x)
}

activation_derivative_funcs = {
    'sigmoid': lambda x: x * (1 - x),
    'relu': lambda x: (x > 0).astype(float),
}