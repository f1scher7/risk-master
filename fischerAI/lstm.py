import numpy as np
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import plot_losses, save_nn_model
from fischerAI.utils.weights_initialization import xavier_init_for_lstm
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class LSTM:

    def __init__(self, input_sequence_length, hidden_neurons, epochs=5000, learning_rate=0.01, initial_state=None):
        self.input_sequence_length = input_sequence_length
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.forget_gate_weights, self.input_gate_weights, self.candidate_gate_weights, self.output_gate_weights = xavier_init_for_lstm(self.input_sequence_length, self.hidden_neurons)

        self.forget_gate_bias = np.zeros((1, self.hidden_neurons))
        self.input_gate_bias = np.zeros((1, self.hidden_neurons))
        self.candidate_gate_bias = np.zeros((1, self.hidden_neurons))
        self.output_gate_bias = np.zeros((1, self.hidden_neurons))

        if initial_state is None:
            self.hidden_state = np.zeros((1, hidden_neurons))
            self.cell_state = np.zeros((1, hidden_neurons))
        else:
            self.hidden_state, self.cell_state = initial_state

        self.hidden_states = None
        self.cell_states = None
        self.forget_gates = None
        self.input_gates = None
        self.candidate_gates = None
        self.output_gates = None
        self.combined_inputs = None


    # def train(self, x, y, verbose=True):
    #     losses = []
    #
    #     for epoch in range(self.epochs):
    #         epoch_loss = 0
    #
    #         for i in range(len(x)):
    #             final_hidden, _, _ = self.forward_propagation(x[i])
    #             self.back_propagation_through_time(y[i])
    #
    #             loss = cost_func(y[i], final_hidden, 'mse')
    #             epoch_loss += loss
    #
    #         avg_loss = epoch_loss / len(x)
    #         losses.append(avg_loss)
    #
    #         if verbose and epoch % 10 == 0:
    #             print(f'Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}')
    #
    #     plot_losses(losses, 'mse')
    #
    #     return losses


    def forward_propagation(self, sequence):
        sequence_length = sequence.shape[0]

        self.reset_memory(sequence_length)

        for t in range(sequence_length):
            current_input = sequence[t].reshape(1, -1)

            combined = np.concatenate((self.hidden_state, current_input), axis=1)

            forget_gate = activation_func(np.dot(combined, self.forget_gate_weights) + self.forget_gate_bias, 'sigmoid')
            input_gate = activation_func(np.dot(combined, self.input_gate_weights) + self.input_gate_bias, 'sigmoid')
            candidate_gate = activation_func(np.dot(combined, self.candidate_gate_weights) + self.candidate_gate_bias, 'tanh')
            output_gate = activation_func(np.dot(combined, self.output_gate_weights) + self.output_gate_bias, 'sigmoid')

            cell_state = forget_gate * self.cell_state + input_gate * candidate_gate
            hidden_state = output_gate * activation_func(cell_state, 'tanh')

            self.hidden_states[t] = hidden_state
            self.cell_states[t] = cell_state
            self.forget_gates[t] = forget_gate
            self.input_gates[t] = input_gate
            self.candidate_gates[t] = candidate_gate
            self.output_gates[t] = output_gate
            self.combined_inputs[t] = combined

            self.hidden_state = hidden_state
            self.cell_state = cell_state

        return self.hidden_states, self.cell_states


    def back_propagation_through_time(self, target_sequence):
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
                d_hidden_state = self.hidden_states[t] - target_sequence[t]
            else:
                d_hidden_state = d_next_hidden_state

            current_cell_state = self.cell_states[t].reshape(1, -1)
            forget_gate = self.forget_gates[t].reshape(1, -1)
            input_gate = self.input_gates[t].reshape(1, -1)
            candidate_gate = self.candidate_gates[t].reshape(1, -1)
            output_gate = self.output_gates[t].reshape(1, -1)
            combined = self.combined_inputs[t].reshape(1, -1)

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


    def reset_memory(self, sequence_length):
        self.hidden_states = np.zeros((sequence_length, self.hidden_neurons))
        self.cell_states = np.zeros((sequence_length, self.hidden_neurons))
        self.forget_gates = np.zeros((sequence_length, self.hidden_neurons))
        self.input_gates = np.zeros((sequence_length, self.hidden_neurons))
        self.candidate_gates = np.zeros((sequence_length, self.hidden_neurons))
        self.output_gates = np.zeros((sequence_length, self.hidden_neurons))
        self.combined_inputs = np.zeros((sequence_length, self.input_sequence_length + self.hidden_neurons))


    def save_model(self, loss_after_training, investment_symbol):
        model_info = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_neurons": self.hidden_neurons,
            "loss": loss_after_training,

            "forget_gate_weights": self.forget_gate_weights,
            "input_gate_weights": self.input_gate_weights,
            "candidate_gate_weights": self.candidate_gate_weights,
            "output_gate_weights": self.output_gate_weights,

            "forget_gate_bias": self.forget_gate_bias,
            "input_gate_bias": self.input_gate_bias,
            "candidate_gate_bias": self.candidate_gate_bias,
            "output_gate_bias": self.output_gate_bias,
        }

        save_nn_model(model_info, PRICE_PREDICTION_SAVED_MODELS_PATH, investment_symbol.value, 'lstm_model')