import numpy as np


def cost_func(y_true, y_pred, cost_func_name):
    return cost_funcs[cost_func_name](y_true, y_pred)


def cost_derivative_func(y_true, y_pred, cost_func_name):
    return cost_derivative_funcs[cost_func_name](y_true, y_pred)


cost_funcs = {
    'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
}

cost_derivative_funcs = {
    'mse': lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.shape[0]
}