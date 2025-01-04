import numpy as np
from time import perf_counter
from fischerAI.lstm import LSTM
from fischerAI.dense_nn import DenseNN
from fischerAI.utils.cost import cost_func
from fischerAI.utils.fischerAI_utils import plot_losses
from fischerAI.utils.fischerAI_utils import display_training_time


class PricePredictionAI:

    def __init__(self, batch_size, sequence_length, input_sequence_length, hidden_lstm_neurons, hidden_dense_neurons, epochs, learning_rate_lstm, learning_rate_dense):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_sequence_length = input_sequence_length
        self.hidden_lstm_neurons = hidden_lstm_neurons
        self.hidden_dense_neurons = hidden_dense_neurons
        self.epochs = epochs
        self.learning_rate_lstm = learning_rate_lstm
        self.learning_rate_dense = learning_rate_dense

        self.lstm = None
        self.dense_nn = None


    def build_model(self):
        self.lstm = LSTM(
            input_sequence_length=self.input_sequence_length, hidden_neurons=self.hidden_lstm_neurons,
            epochs=self.epochs, learning_rate=self.learning_rate_lstm
        )

        self.dense_nn = DenseNN(
            x=np.zeros((self.batch_size, self.sequence_length, self.hidden_lstm_neurons)), # to be updated in train()
            y=np.zeros((self.batch_size, self.sequence_length, 1)), # to be updated in train()
            hidden1_neurons=self.hidden_dense_neurons,
            epochs=self.epochs,
            learning_rate=self.learning_rate_dense
        )


    def train(self, train_sequences, train_target_sequences):
        training_start_time = perf_counter()

        lstm_losses = []

        for epoch in range(self.epochs):
            lstm_epoch_loss = 0

            lstm_features = []
            inc_for_lstm = 0
            train_sequences_length = train_sequences.shape[0]

            for sequence, target_sequence in zip(train_sequences, train_target_sequences):
                final_hidden_states, _ = self.lstm.forward_propagation(sequence)

                self.lstm.back_propagation_through_time(target_sequence)

                lstm_features.append(final_hidden_states)

                lstm_loss = cost_func(target_sequence, final_hidden_states, 'mse')
                lstm_epoch_loss += lstm_loss

                inc_for_lstm += 1
                if inc_for_lstm % 1000 == 0:
                    print("Sequences processed in LSTM: ", inc_for_lstm, '/', train_sequences_length)


            lstm_features = np.array(lstm_features)
            lstm_features_reshaped = lstm_features.reshape(self.batch_size, self.sequence_length, self.hidden_lstm_neurons)

            self.dense_nn.x = lstm_features_reshaped
            self.dense_nn.y = train_target_sequences

            hidden1_activated_arr, output_activated_arr = self.dense_nn.forward_propagation()
            self.dense_nn.back_propagation(hidden1_activated_arr, output_activated_arr)

            lstm_avg_loss = lstm_epoch_loss / len(train_sequences)
            lstm_losses.append(lstm_avg_loss)

            self.display_training_info(epoch, lstm_avg_loss)


        display_training_time(training_start_time)

        self.lstm.save_model()
        self.dense_nn.save_model()

        plot_losses(lstm_losses, 'MSE', 'LSTM')
        plot_losses(self.dense_nn.mse_values, 'MSE', 'Dense')

        return lstm_losses


    def display_training_info(self, epoch, lstm_avg_loss):
        print("=" * 41)
        print("Epoch  " + str(epoch + 1) + " / " + str(self.epochs))
        print("LSTM Loss: " + "{:.6f}".format(lstm_avg_loss))
        print("Dense Loss: " + "{:.6f}".format(self.dense_nn.mse_values[epoch]))
        print("=" * 41)
