import numpy as np
from fischerAI.utils.weights_initialization import he_init
from fischerAI.utils.activation_funcs import activation_func, activation_derivative_func
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import plot_losses


# Hidden layer type - Dense
# Weights init - He
class DenseNN:

    def __init__(self, x, y, hidden1_neurons, epochs, learning_rate):
        self.x = x
        self.y = y
        self.hidden1_neurons = hidden1_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden1_activated = None
        self.output_activated = None
        self.mse_values = []

        self.input_size = self.x.shape[1]
        self.output_size = self.y.shape[1]

        self.input_to_hidden1_weights = he_init(self.input_size, self.hidden1_neurons)
        self.hidden1_to_output_weights = he_init(self.hidden1_neurons, self.output_size)

        self.bias_hidden1_weights = np.zeros((1, self.hidden1_neurons))
        self.bias_output_weights = np.zeros((1, self.output_size))


    def train(self):
        for epoch in self.epochs:
            self.forward_propagation()
            self.back_propagation()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} - MSE: {self.mse_values[-1]}")

        plot_losses(self.mse_values, 'mse')


    def forward_propagation(self):
        hidden1_pre_activation = np.dot(self.x, self.input_to_hidden1_weights) + self.bias_hidden1_weights
        self.hidden1_activated = activation_func(hidden1_pre_activation, 'relu')

        output_pre_activation = np.dot(self.hidden1_activated, self.hidden1_to_output_weights) + self.bias_output_weights
        self.output_activated = activation_func(output_pre_activation, 'relu')


    def back_propagation(self):
        self.mse_values.append(cost_func(self.y, self.output_activated, 'mse'))

        output_error = self.y - self.output_activated
        output_gradient = output_error * activation_derivative_func(self.output_activated, 'relu')

        hidden1_error = np.dot(output_gradient, self.hidden1_to_output_weights.T)
        hidden1_gradient = hidden1_error * activation_derivative_func(self.hidden1_activated, 'relu')

        self.hidden1_to_output_weights += self.learning_rate * np.dot(self.hidden1_activated.T, output_gradient)
        self.input_to_hidden1_weights += self.learning_rate * np.dot(self.x.T, hidden1_gradient)

        self.bias_output_weights += self.learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        self.bias_hidden1_weights += self.learning_rate * np.sum(hidden1_gradient, axis=0, keepdims=True)

