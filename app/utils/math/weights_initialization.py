import numpy as np


np.random.seed(42)


def weights_initialization_func(shape, func_name):
    return weights_initialization_funcs[func_name](shape)


weights_initialization_funcs = {
    'random': lambda shape: np.random.uniform(low=-1, high=1, size=shape),
    'normal': lambda shape: np.random.randn(*shape),
    'xavier': lambda shape: np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1])),
    'he': lambda shape: np.random.randn(*shape) * np.sqrt(2 / shape[0]),
}
