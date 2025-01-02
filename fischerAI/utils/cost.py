import numba
import numpy as np


@numba.jit(nopython=True)
def mse(y_true, y_pred):
    return  np.mean(np.square(y_pred - y_true))

@numba.jit(nopython=True)
def mse_derivative(y_true, y_pred):
    return 2. * (y_pred - y_true) / y_true.shape[0]


@numba.jit(nopython=True)
def cost_func(y_true, y_pred, cost_func_name):
    if cost_func_name == 'mse':
        return mse(y_true, y_pred)

@numba.jit(nopython=True)
def cost_derivative_func(y_true, y_pred, cost_func_name):
    if cost_func_name == 'mse':
        return mse_derivative(y_true, y_pred)
