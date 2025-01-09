import numpy as np
from fischerAI.utils.fischerAI_utils import save_nn_model
from fischerAI.utils.weights_initialization import xavier_init_for_lstm
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class LSTM:

    def __init__(self, input_sequence_length, sequences_min, sequences_max, hidden_layers_neurons, epochs=100, learning_rate=0.01, momentum=0.9):
        self.input_sequence = None
        self.input_sequence_length = input_sequence_length

        self.sequence_min = sequences_min
        self.sequence_max = sequences_max

        self.hidden_layers_neurons = hidden_layers_neurons
        self.num_layers = len(hidden_layers_neurons)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.layers = []

        input_size = input_sequence_length
        for i, hidden_size in enumerate(hidden_layers_neurons):
            layer = {
                'forget_gate_weights': (xavier_init_for_lstm(input_size, hidden_size))[0],
                'input_gate_weights': (xavier_init_for_lstm(input_size, hidden_size))[1],
                'candidate_gate_weights': (xavier_init_for_lstm(input_size, hidden_size))[2],
                'output_gate_weights': (xavier_init_for_lstm(input_size, hidden_size))[3],
                'forget_gate_bias': np.ones((1, hidden_size)),
                'input_gate_bias': np.zeros((1, hidden_size)),
                'candidate_gate_bias': np.zeros((1, hidden_size)),
                'output_gate_bias': np.zeros((1, hidden_size)),
                'hidden_state': np.zeros((1, hidden_size)),
                'cell_state': np.zeros((1, hidden_size)),
                'velocity': {}
            }

            layer['velocity'] = {
                'forget_gate_weights': np.zeros_like(layer['forget_gate_weights']),
                'input_gate_weights': np.zeros_like(layer['input_gate_weights']),
                'candidate_gate_weights': np.zeros_like(layer['candidate_gate_weights']),
                'output_gate_weights': np.zeros_like(layer['output_gate_weights']),
                'forget_gate_bias': np.zeros_like(layer['forget_gate_bias']),
                'input_gate_bias': np.zeros_like(layer['input_gate_bias']),
                'candidate_gate_bias': np.zeros_like(layer['candidate_gate_bias']),
                'output_gate_bias': np.zeros_like(layer['output_gate_bias'])
            }

            self.layers.append(layer)
            input_size = hidden_size


    def forward_propagation(self, sequence):
        self.input_sequence = sequence

        sequence_length = sequence.shape[0]
        layer_outputs = []

        for layer in self.layers:
            layer['hidden_states'] = np.zeros((sequence_length, layer['hidden_state'].shape[1]))
            layer['cell_states'] = np.zeros((sequence_length, layer['cell_state'].shape[1]))
            layer['forget_gates'] = np.zeros((sequence_length, layer['hidden_state'].shape[1]))
            layer['input_gates'] = np.zeros((sequence_length, layer['hidden_state'].shape[1]))
            layer['candidate_gates'] = np.zeros((sequence_length, layer['hidden_state'].shape[1]))
            layer['output_gates'] = np.zeros((sequence_length, layer['hidden_state'].shape[1]))

        current_input = sequence

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = []

            for t in range(sequence_length):
                current_time_input = current_input[t].reshape(1, -1)

                combined = np.concatenate((layer['hidden_state'], current_time_input), axis=1)

                forget_gate = activation_func(np.dot(combined, layer['forget_gate_weights']) + layer['forget_gate_bias'], 'sigmoid')
                input_gate = activation_func(np.dot(combined, layer['input_gate_weights']) + layer['input_gate_bias'], 'sigmoid')
                candidate_gate = activation_func(np.dot(combined, layer['candidate_gate_weights']) + layer['candidate_gate_bias'], 'tanh')
                output_gate = activation_func(np.dot(combined, layer['output_gate_weights']) + layer['output_gate_bias'], 'sigmoid')

                cell_state = forget_gate * layer['cell_state'] + input_gate * candidate_gate
                hidden_state = output_gate * activation_func(cell_state, 'tanh')

                layer['hidden_states'][t] = hidden_state
                layer['cell_states'][t] = cell_state
                layer['forget_gates'][t] = forget_gate
                layer['input_gates'][t] = input_gate
                layer['candidate_gates'][t] = candidate_gate
                layer['output_gates'][t] = output_gate

                layer['hidden_state'] = hidden_state
                layer['cell_state'] = cell_state

                hidden_states.append(hidden_state)

            current_input = np.array(hidden_states).reshape(sequence_length, -1)
            layer_outputs.append(current_input)

        return layer_outputs[-1], [layer['cell_states'] for layer in self.layers]


    def back_propagation_through_time(self, target_sequence):
        sequence_length = len(self.layers[0]['hidden_states'])

        for layer_idx in reversed(range(self.num_layers)):
            layer = self.layers[layer_idx]

            d_next_hidden = np.zeros_like(layer['hidden_states'][0])
            d_next_cell = np.zeros_like(layer['cell_states'][0])

            d_weights = {
                'forget_gate': np.zeros_like(layer['forget_gate_weights']),
                'input_gate': np.zeros_like(layer['input_gate_weights']),
                'candidate_gate': np.zeros_like(layer['candidate_gate_weights']),
                'output_gate': np.zeros_like(layer['output_gate_weights']),
                'forget_bias': np.zeros_like(layer['forget_gate_bias']),
                'input_bias': np.zeros_like(layer['input_gate_bias']),
                'candidate_bias': np.zeros_like(layer['candidate_gate_bias']),
                'output_bias': np.zeros_like(layer['output_gate_bias'])
            }

            for t in reversed(range(sequence_length)):
                current_cell_state = layer['cell_states'][t].reshape(1, -1)
                forget_gate = layer['forget_gates'][t].reshape(1, -1)
                input_gate = layer['input_gates'][t].reshape(1, -1)
                candidate_gate = layer['candidate_gates'][t].reshape(1, -1)
                output_gate = layer['output_gates'][t].reshape(1, -1)

                if layer_idx == self.num_layers - 1 and t == sequence_length - 1:
                    d_hidden = layer['hidden_states'][t] - target_sequence[t]
                else:
                    d_hidden = d_next_hidden

                d_output = d_hidden * activation_func(current_cell_state, 'tanh') * \
                           activation_derivative_func(output_gate, 'sigmoid')

                d_cell = d_hidden * output_gate * activation_derivative_func(current_cell_state, 'tanh') + \
                         d_next_cell

                if t > 0:
                    prev_cell_state = layer['cell_states'][t - 1]
                else:
                    prev_cell_state = np.zeros_like(current_cell_state)

                d_forget = d_cell * prev_cell_state * activation_derivative_func(forget_gate, 'sigmoid')
                d_input = d_cell * candidate_gate * activation_derivative_func(input_gate, 'sigmoid')
                d_candidate = d_cell * input_gate * activation_derivative_func(candidate_gate, 'tanh')

                if t > 0:
                    prev_hidden = layer['hidden_states'][t - 1]
                else:
                    prev_hidden = np.zeros((1, layer['hidden_state'].shape[1]))

                if layer_idx == 0:
                    current_input = self.input_sequence[t].reshape(1, -1)
                else:
                    current_input = self.layers[layer_idx - 1]['hidden_states'][t]

                prev_hidden = prev_hidden.reshape(1, -1) if prev_hidden.ndim == 1 else prev_hidden
                current_input = current_input.reshape(1, -1) if current_input.ndim == 1 else current_input

                combined_input = np.concatenate((prev_hidden, current_input), axis=1)

                d_weights['forget_gate'] += np.dot(combined_input.T, d_forget)
                d_weights['input_gate'] += np.dot(combined_input.T, d_input)
                d_weights['candidate_gate'] += np.dot(combined_input.T, d_candidate)
                d_weights['output_gate'] += np.dot(combined_input.T, d_output)

                d_weights['forget_bias'] += d_forget
                d_weights['input_bias'] += d_input
                d_weights['candidate_bias'] += d_candidate
                d_weights['output_bias'] += d_output

                if t > 0:
                    hidden_size = layer['hidden_state'].shape[1]
                    d_combined = (
                            np.dot(d_forget, layer['forget_gate_weights'].T) +
                            np.dot(d_input, layer['input_gate_weights'].T) +
                            np.dot(d_candidate, layer['candidate_gate_weights'].T) +
                            np.dot(d_output, layer['output_gate_weights'].T)
                    )
                    d_next_hidden = d_combined[:, :hidden_size]
                    d_next_cell = d_cell * forget_gate

                if layer_idx > 0:
                    d_prev_layer = d_combined[:, hidden_size:]
                    self.layers[layer_idx - 1]['d_hidden_from_next'] = d_prev_layer

            for gate in ['forget_gate', 'input_gate', 'candidate_gate', 'output_gate']:
                weights_key = f'{gate}_weights'
                bias_key = f'{gate}_bias'

                clip_threshold = 1.0

                d_weights[gate] = np.clip(d_weights[gate], -clip_threshold, clip_threshold)
                d_weights[f'{gate[:-5]}_bias'] = np.clip(d_weights[f'{gate[:-5]}_bias'], -clip_threshold, clip_threshold)

                layer['velocity'][weights_key] = (
                        self.momentum * layer['velocity'][weights_key] +
                        self.learning_rate * d_weights[gate]
                )
                layer[weights_key] -= layer['velocity'][weights_key]

                layer['velocity'][bias_key] = (
                        self.momentum * layer['velocity'][bias_key] +
                        self.learning_rate * d_weights[f'{gate[:-5]}_bias']
                )
                layer[bias_key] -= layer['velocity'][bias_key]


    def reset_states(self):
        for layer in self.layers:
            layer['hidden_state'] = np.zeros_like(layer['hidden_state'])
            layer['cell_state'] = np.zeros_like(layer['cell_state'])


    def save_model(self, loss_after_training, investment_symbol):
        model_info = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_layers_neurons": self.hidden_layers_neurons,
            "loss": loss_after_training,

            "sequences_min": self.sequence_min,
            "sequences_max": self.sequence_max,

            "layers": [{
                "forget_gate_weights": layer['forget_gate_weights'],
                "input_gate_weights": layer['input_gate_weights'],
                "candidate_gate_weights": layer['candidate_gate_weights'],
                "output_gate_weights": layer['output_gate_weights'],
                "forget_gate_bias": layer['forget_gate_bias'],
                "input_gate_bias": layer['input_gate_bias'],
                "candidate_gate_bias": layer['candidate_gate_bias'],
                "output_gate_bias": layer['output_gate_bias']} for layer in self.layers],
        }

        save_nn_model(model_info, PRICE_PREDICTION_SAVED_MODELS_PATH, investment_symbol.value, 'multi_layer_lstm_model')