import numpy as np
from fischerAI.utils.weights_initialization import he_init
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import save_nn_model
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class DenseNN:

    def __init__(self, x, y, hidden1_neurons, epochs=10, learning_rate=0.0000001):
        self.x = x
        self.y = y
        self.hidden1_neurons = hidden1_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.input_size = self.x.shape[2]
        self.output_size = self.y.shape[2]
        self.batch_size = self.x.shape[0]
        self.target_sequence_length = self.y.shape[1]

        self.input_to_hidden1_weights = he_init(self.input_size, self.hidden1_neurons, False)
        self.hidden1_to_output_weights = he_init(self.hidden1_neurons, self.output_size, False)

        self.hidden1_bias = np.zeros((1, self.hidden1_neurons))
        self.output_bias = np.zeros((1, self.output_size))

        self.mse_values = []


    def forward_propagation(self):
        hidden1_activated_arr = np.zeros((self.batch_size, self.target_sequence_length, self.hidden1_neurons))
        output_activated_arr = np.zeros((self.batch_size, self.target_sequence_length, self.output_size))

        for t in range(self.target_sequence_length):
            x_t = self.x[:, t, :]

            hidden1_pre_activation = np.dot(x_t, self.input_to_hidden1_weights) + self.hidden1_bias
            hidden1_activated = activation_func(hidden1_pre_activation, 'leaky_relu')

            output_pre_activation = np.dot(hidden1_activated, self.hidden1_to_output_weights) + self.output_bias
            output_activated = output_pre_activation # linear act func

            hidden1_activated_arr[:, t, :] = hidden1_activated
            output_activated_arr[:, t, :] = output_activated

        return hidden1_activated_arr, output_activated_arr


    def back_propagation(self, hidden1_activated_arr, output_activated_arr):
        mse_values_per_t = []

        for t in range(self.target_sequence_length):
            mse_values_per_t.append(cost_func(self.y[:, t, :], output_activated_arr[:, t, :], 'mse'))

            output_error = self.y[:, t, :] - output_activated_arr[:, t, :]
            output_gradient = output_error # derivative for linear act func is 1

            hidden1_error = np.dot(output_gradient, self.hidden1_to_output_weights.T)
            hidden1_gradient = hidden1_error * activation_derivative_func(hidden1_activated_arr[:, t, :], 'leaky_relu')

            self.hidden1_to_output_weights += self.learning_rate * np.dot(hidden1_activated_arr[:, t, :].T, output_gradient)
            self.input_to_hidden1_weights += self.learning_rate * np.dot(self.x[:, t, :].T, hidden1_gradient)

            self.output_bias += self.learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
            self.hidden1_bias += self.learning_rate * np.sum(hidden1_gradient, axis=0, keepdims=True)

            # print("Dense avg weights:", np.mean(self.input_to_hidden1_weights), np.mean(self.hidden1_to_output_weights))

        self.mse_values.append(np.mean(mse_values_per_t))


    def save_model(self, loss_after_training, investment_symbol):
        model_info = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden1_neurons": self.hidden1_neurons,
            "loss": loss_after_training,

            "input_to_hidden1_weights": self.input_to_hidden1_weights,
            "hidden1_to_output_weights": self.hidden1_to_output_weights,

            "hidden1_bias": self.hidden1_bias,
            "output_bias": self.output_bias,
        }

        save_nn_model(model_info, PRICE_PREDICTION_SAVED_MODELS_PATH, investment_symbol.value, 'dense_nn_model')
