import numpy as np

np.random.seed(42)


def he_init(current_layer_size, next_layer_size):
    return np.random.randn(current_layer_size, next_layer_size) * np.sqrt(2. / current_layer_size)


def xavier_init_for_lstm(input_size, hidden_size):
    combined_size = input_size + hidden_size

    forget_gate_weight = xavier_init(combined_size, hidden_size)
    input_gate_weight = xavier_init(combined_size, hidden_size)
    output_gate_weight = xavier_init(combined_size, hidden_size)
    client_gate_weight = xavier_init(combined_size, hidden_size)

    return forget_gate_weight, input_gate_weight, output_gate_weight, client_gate_weight


def xavier_init(current_layer_size, next_layer_size):
    limit = np.sqrt(6. / (current_layer_size + next_layer_size)) # we have 6. because its the best value for sigmoid and tanh
    return np.random.uniform(low=-limit, high=-limit, size=(current_layer_size, next_layer_size))


# def weights_initialization_func(shape, func_name):
#     return weights_initialization_funcs[func_name](shape)
#
#
# weights_initialization_funcs = {
#     'random': lambda shape: np.random.uniform(low=-1, high=1, size=shape),
#     'normal': lambda shape: np.random.randn(*shape),
#     'xavier': lambda shape: np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1])),
#     'he': lambda shape: np.random.randn(*shape) * np.sqrt(2 / shape[0]),
# }
