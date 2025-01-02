import numpy as np
from datetime import datetime
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import plot_losses
from fischerAI.utils.weights_initialization import xavier_init_for_lstm
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class LSTM:

    def __init__(self, input_size, hidden_size, epochs=5000, learning_rate=0.01, initial_state=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.forget_gate_weights, self.input_gate_weights, self.candidate_gate_weights, self.output_gate_weights = xavier_init_for_lstm(self.input_size, self.hidden_size)

        self.forget_gate_bias = np.zeros((1, self.hidden_size))
        self.input_gate_bias = np.zeros((1, self.hidden_size))
        self.candidate_gate_bias = np.zeros((1, self.hidden_size))
        self.output_gate_bias = np.zeros((1, self.hidden_size))

        if initial_state is None:
            self.hidden_state = np.zeros((1, hidden_size))
            self.cell_state = np.zeros((1, hidden_size))
        else:
            self.hidden_state, self.cell_state = initial_state

        self.hidden_states = None
        self.cell_states = None
        self.forget_gates = None
        self.input_gates = None
        self.candidate_gates = None
        self.output_gates = None
        self.combined_inputs = None

        self.reset_memory()


    def train(self, x, y, verbose=True):
        losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0

            for i in range(len(x)):
                final_hidden, _, _ = self.forward_propagation(x[i])
                self.back_propagation_through_time(y[i])

                loss = cost_func(y[i], final_hidden, 'mse')
                epoch_loss += loss

            avg_loss = epoch_loss / len(x)
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}')

        plot_losses(losses, 'mse')

        return losses


    def forward_propagation(self, input_sequence):
        self.reset_memory()

        for t in range(len(input_sequence)):
            current_input = input_sequence[t].reshape(1, -1)

            combined = np.concatenate((self.hidden_state, current_input), axis=1)

            forget_gate = activation_func(np.dot(combined, self.forget_gate_weights) + self.forget_gate_bias, 'sigmoid')
            input_gate = activation_func(np.dot(combined, self.input_gate_weights) + self.input_gate_bias, 'sigmoid')
            candidate_gate = activation_func(np.dot(combined, self.candidate_gate_weights) + self.candidate_gate_bias, 'tanh')
            output_gate = activation_func(np.dot(combined, self.output_gate_weights) + self.output_gate_bias, 'sigmoid')

            cell_state = forget_gate * self.cell_state + input_gate * candidate_gate
            hidden_state = output_gate * activation_func(cell_state, 'tanh')

            self.hidden_states.append(hidden_state)
            self.cell_states.append(cell_state)
            self.forget_gates.append(forget_gate)
            self.input_gates.append(input_gate)
            self.candidate_gates.append(candidate_gate)
            self.output_gates.append(output_gate)
            self.combined_inputs.append(combined)

            self.hidden_state = hidden_state
            self.cell_state = cell_state

        return self.hidden_state, self.hidden_states, self.cell_states


    def back_propagation_through_time(self, target):
        sequence_length = len(self.hidden_states)

        d_forget_gate_weights = np.zeros_like(self.forget_gate_weights)
        d_input_gate_weights = np.zeros_like(self.input_gate_weights)
        d_candidate_gate_weights = np.zeros_like(self.candidate_gate_weights)
        d_output_gate_weights = np.zeros_like(self.output_gate_weights)

        d_forget_gate_bias = np.zeros_like(self.forget_gate_bias)
        d_input_gate_bias = np.zeros_like(self.input_gate_bias)
        d_candidate_gate_bias = np.zeros_like(self.candidate_gate_bias)
        d_output_gate_bias = np.zeros_like(self.output_gate_bias)

        d_next_hidden_state = np.zeros_like(self.hidden_states[0])
        d_next_cell_state = np.zeros_like(self.cell_states[0])

        for t in reversed(range(sequence_length)):
            if t == sequence_length - 1:
                d_hidden_state = self.hidden_states[t] - target
            else:
                d_hidden_state = d_next_hidden_state

            current_cell_state = self.cell_states[t]
            forget_gate = self.forget_gates[t]
            input_gate = self.input_gates[t]
            candidate_gate = self.candidate_gates[t]
            output_gate = self.output_gates[t]
            combined = self.combined_inputs[t]

            d_output_gate = d_hidden_state * activation_func(current_cell_state, 'tanh') * activation_derivative_func(output_gate, 'sigmoid')
            d_cell_state = d_hidden_state * output_gate * activation_derivative_func(current_cell_state, 'tanh') + d_next_cell_state
            d_forget_gate = d_cell_state * self.cell_states[t - 1 if t > 0 else 0] * activation_derivative_func(forget_gate, 'sigmoid')
            d_input_gate = d_cell_state * candidate_gate * activation_derivative_func(input_gate, 'sigmoid')
            d_candidate_gate = d_cell_state * input_gate * activation_derivative_func(candidate_gate, 'tanh')

            d_output_gate_weights += np.dot(combined.T, d_output_gate)
            d_forget_gate_weights += np.dot(combined.T, d_forget_gate)
            d_input_gate_weights += np.dot(combined.T, d_input_gate)
            d_candidate_gate_weights += np.dot(combined.T, d_candidate_gate)

            d_output_gate_bias += d_output_gate
            d_forget_gate_bias += d_forget_gate
            d_input_gate_bias += d_input_gate
            d_candidate_gate_bias += d_candidate_gate

            if t > 0:
                hidden_state_size = self.hidden_state.shape[0]

                d_next_hidden_state_forget = np.dot(d_forget_gate, self.forget_gate_weights[:hidden_state_size, :].T)
                d_next_hidden_state_input = np.dot(d_input_gate, self.input_gate_weights[:hidden_state_size, :].T)
                d_next_hidden_state_candidate = np.dot(d_candidate_gate, self.candidate_gate_weights[:hidden_state_size, :].T)
                d_next_hidden_state_output = np.dot(d_output_gate, self.output_gate_weights[:hidden_state_size, :].T)

                d_next_hidden_state = d_next_hidden_state_forget + d_next_hidden_state_input + d_next_hidden_state_candidate + d_next_hidden_state_output

            # Prepare next cell gradient for next step
            d_next_cell_state = d_cell_state * forget_gate

        gradients = [
            d_forget_gate_weights, d_input_gate_weights, d_candidate_gate_weights, d_output_gate_weights,
            d_forget_gate_bias, d_input_gate_bias, d_candidate_gate_bias, d_output_gate_bias
        ]

        normalized_gradients = self.gradients_clipping_by_norm(gradients)

        self.forget_gate_weights -= self.learning_rate * normalized_gradients[0]
        self.input_gate_weights -= self.learning_rate * normalized_gradients[1]
        self.candidate_gate_weights -= self.learning_rate * normalized_gradients[2]
        self.output_gate_weights -= self.learning_rate * normalized_gradients[3]

        self.forget_gate_bias -= self.learning_rate * normalized_gradients[4]
        self.input_gate_bias -= self.learning_rate * normalized_gradients[5]
        self.candidate_gate_bias -= self.learning_rate * normalized_gradients[6]
        self.output_gate_bias -= self.learning_rate * normalized_gradients[7]


    def gradients_clipping_by_norm(self, gradients, threshold=5.0):
        total_norm = np.sqrt(sum(np.sum(np.square(gradient)) for gradient in gradients))
        scaling_factor = threshold / max(total_norm, threshold)
        return [gradient * scaling_factor for gradient in gradients]


    def reset_memory(self):
        self.hidden_states = []
        self.cell_states = []
        self.forget_gates = []
        self.input_gates = []
        self.candidate_gates = []
        self.output_gates = []
        self.combined_inputs = []
