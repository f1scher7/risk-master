import numpy as np
from fischerAI.lstm import LSTM
from fischerAI.dense_nn import DenseNN
from fischerAI.utils.cost import cost_func


class PricePredictionAI:

    def __init__(self, sequence_length=300, hidden_lstm_neurons=64, hidden_dense_neurons=32, epochs=27, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.hidden_lstm_neurons = hidden_lstm_neurons
        self.hidden_dense_neurons = hidden_dense_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.lstm = None
        self.dense_nn = None


    def build_model(self, input_sequence_size):
        self.lstm = LSTM(
            input_sequence_size=input_sequence_size, hidden_size=self.hidden_lstm_neurons,
            epochs=self.epochs, learning_rate=self.learning_rate
        )

        self.dense_nn = DenseNN(
            x=np.zeros((1, self.hidden_lstm_neurons)), # To be updated later
            y=np.zeros((1, 1)), # To be updated later
            hidden1_neurons=self.hidden_dense_neurons,
            epochs=self.epochs,
            learning_rate=self.learning_rate
        )


    def train(self, train_sequences, train_target_sequences):
        losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0

            lstm_features = []
            inc_for_lstm = 0
            train_sequences_length = train_sequences.shape[0]

            for sequence, target_sequence in zip(train_sequences, train_target_sequences):
                final_hidden_states, _ = self.lstm.forward_propagation(sequence)

                self.lstm.back_propagation_through_time(target_sequence)

                lstm_features.append(final_hidden_states)

                loss = cost_func(target_sequence, final_hidden_states, 'mse')
                epoch_loss += loss

                inc_for_lstm += 1
                print(inc_for_lstm, '/', train_sequences_length)


            lstm_features = np.vstack(lstm_features)

            self.dense_nn.x = lstm_features
            self.dense_nn.y = train_target_sequences

            print()

            self.dense_nn.forward_propagation()
            self.dense_nn.back_propagation()

            avg_loss = epoch_loss / len(train_sequences)
            print("Epoch " + str(epoch) + "/" + str(self.epochs) + " - LSTM Loss: " + "{:.6f}".format(avg_loss))
            losses.append(avg_loss)


        self.lstm.save_model()
        self.dense_nn.save_model()

        return losses
