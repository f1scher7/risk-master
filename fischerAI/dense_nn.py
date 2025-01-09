import numpy as np
from fischerAI.utils.weights_initialization import he_init
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import save_nn_model
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class DenseNN:
    def __init__(self, x, y, target_sequences_min, target_sequences_max, hidden_layers_neurons, epochs=1000, learning_rate=0.0000001, momentum=0.9):
        self.x = x
        self.y = y
        self.target_sequences_min = target_sequences_min
        self.target_sequences_max = target_sequences_max
        self.hidden_layers_neurons = hidden_layers_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.input_size = self.x.shape[2]
        self.output_size = self.y.shape[2]
        self.batch_size = self.x.shape[0]
        self.target_sequence_length = self.y.shape[1]

        self.weights = []
        self.biases = []
        self.velocities_weights = []
        self.velocities_biases = []

        previous_layer_size = self.input_size

        for neurons in self.hidden_layers_neurons:
            self.weights.append(he_init(previous_layer_size, neurons, False))
            self.biases.append(np.zeros((1, neurons)))
            self.velocities_weights.append(np.zeros((previous_layer_size, neurons)))
            self.velocities_biases.append(np.zeros((1, neurons)))
            previous_layer_size = neurons

        self.weights.append(he_init(previous_layer_size, self.output_size, False))
        self.biases.append(np.zeros((1, self.output_size)))
        self.velocities_weights.append(np.zeros((previous_layer_size, self.output_size)))
        self.velocities_biases.append(np.zeros((1, self.output_size)))

        self.mse_values = []


    def forward_propagation(self):
        activations = [[] for _ in range(len(self.weights))]

        for t in range(self.target_sequence_length):
            layer_input = self.x[:, t, :]

            for i in range(len(self.weights) - 1):
                pre_activation = np.dot(layer_input, self.weights[i]) + self.biases[i]
                layer_input = activation_func(pre_activation, 'leaky_relu')

                if len(activations[i]) == 0:
                    activations[i] = np.zeros((self.batch_size, self.target_sequence_length, layer_input.shape[1]))

                activations[i][:, t, :] = layer_input

            output_pre_activation = np.dot(layer_input, self.weights[-1]) + self.biases[-1]
            output_activated = output_pre_activation

            if len(activations[-1]) == 0:
                activations[-1] = np.zeros((self.batch_size, self.target_sequence_length, output_activated.shape[1]))

            activations[-1][:, t, :] = output_activated

        return activations


    def back_propagation(self, activations):
        mse_values_per_t = []
        weights_gradients = [np.zeros_like(w) for w in self.weights]
        biases_gradients = [np.zeros_like(b) for b in self.biases]

        for t in range(self.target_sequence_length):
            output_error = self.y[:, t, :] - activations[-1][:, t, :]
            mse_values_per_t.append(cost_func(self.y[:, t, :], activations[-1][:, t, :], 'mse'))

            current_gradient = output_error
            for i in range(len(self.weights) - 1, -1, -1):
                if i == len(self.weights) - 1:
                    layer_gradient = current_gradient
                else:
                    current_gradient = np.dot(current_gradient, self.weights[i + 1].T)
                    layer_gradient = current_gradient * activation_derivative_func(activations[i][:, t, :], 'leaky_relu')

                input_data = self.x[:, t, :] if i == 0 else activations[i - 1][:, t, :]
                weights_gradients[i] += np.dot(input_data.T, layer_gradient)
                biases_gradients[i] += np.sum(layer_gradient, axis=0, keepdims=True)

        self.mse_values.append(np.mean(mse_values_per_t))

        clip_threshold = 1.0
        for i in range(len(self.weights)):
            weights_gradients[i] = np.clip(weights_gradients[i], -clip_threshold, clip_threshold)

            self.velocities_weights[i] = self.momentum * self.velocities_weights[i] + weights_gradients[i]
            self.velocities_biases[i] = self.momentum * self.velocities_biases[i] + biases_gradients[i]

            self.weights[i] += self.learning_rate * self.velocities_weights[i]
            self.biases[i] += self.learning_rate * self.velocities_biases[i]


    def save_model(self, loss_after_training, investment_symbol):
        model_info = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_layers_neurons": self.hidden_layers_neurons,
            "loss": loss_after_training,

            "target_sequences_min": self.target_sequences_min,
            "target_sequences_max": self.target_sequences_max,

            "weights": self.weights,
            "biases": self.biases,
        }

        save_nn_model(model_info, PRICE_PREDICTION_SAVED_MODELS_PATH, investment_symbol.value, 'dense_nn_model')