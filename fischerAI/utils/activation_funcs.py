import numba
import numpy as np


@numba.jit(nopython=True)
def relu(x):
    return np.maximum(x, 0)

@numba.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@numba.jit(nopython=True)
def tanh(x):
    return np.tanh(x)


@numba.jit(nopython=True)
def relu_derivative(x):
    return (x > 0) * 1.0

@numba.jit(nopython=True)
def sigmoid_derivative(x):
    return x * (1 - x)

@numba.jit(nopython=True)
def tanh_derivative(x):
    return 1 - x * x


@numba.jit(nopython=True)
def activation_func(x, func_name):
    if func_name == 'relu':
        return relu(x)
    elif func_name == 'sigmoid':
        return sigmoid(x)
    elif func_name == 'tanh':
        return tanh(x)
    else:
        raise ValueError(f"Unknown activation function: {func_name}")

@numba.jit(nopython=True)
def activation_derivative_func(x, func_name):
    if func_name == 'relu':
        return relu_derivative(x)
    elif func_name == 'sigmoid':
        return sigmoid_derivative(x)
    elif func_name == 'tanh':
        return tanh_derivative(x)
    else:
        raise ValueError(f"Unknown derivative function: {func_name}")