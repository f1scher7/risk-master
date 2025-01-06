import numpy as np
from time import perf_counter
from fischerAI.lstm import LSTM
from fischerAI.dense_nn import DenseNN
from fischerAI.utils.cost import cost_func
from fischerAI.utils.input_data_normalization import min_max_denormalization
from fischerAI.utils.fischerAI_utils import plot_losses
from fischerAI.utils.fischerAI_utils import display_training_time
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH


class PricePredictionAI:

    def __init__(self, investment_symbol, batch_size, sequence_length, input_sequence_length, hidden_lstm_neurons, hidden_dense_neurons, epochs, learning_rate_lstm, learning_rate_dense, target_sequences_min, target_sequences_max):
        self.investment_symbol = investment_symbol

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_sequence_length = input_sequence_length

        self.hidden_lstm_neurons = hidden_lstm_neurons
        self.hidden_dense_neurons = hidden_dense_neurons
        self.epochs = epochs
        self.learning_rate_lstm = learning_rate_lstm
        self.learning_rate_dense = learning_rate_dense

        self.target_sequences_min = target_sequences_min
        self.target_sequences_max = target_sequences_max

        self.lstm = None
        self.dense_nn = None


    def build_model(self):
        self.lstm = LSTM(
            input_sequence_length=self.input_sequence_length, hidden_neurons=self.hidden_lstm_neurons,
            epochs=self.epochs, learning_rate=self.learning_rate_lstm,

        )

        self.dense_nn = DenseNN(
            x=np.zeros((self.batch_size, self.sequence_length, self.hidden_lstm_neurons)), # to be updated in train()
            y=np.zeros((self.batch_size, self.sequence_length, 1)), # to be updated in train()
            target_sequences_min=self.target_sequences_min,
            target_sequences_max=self.target_sequences_max,
            hidden1_neurons=self.hidden_dense_neurons,
            epochs=self.epochs,
            learning_rate=self.learning_rate_dense
        )


    def train(self, train_sequences, train_target_sequences):
        training_start_time = perf_counter()

        lstm_losses = []

        for epoch in range(self.epochs):
            lstm_features = []
            lstm_epoch_loss = 0
            inc_for_lstm = 0

            train_sequences_length = train_sequences.shape[0]

            for sequence, target_sequence in zip(train_sequences, train_target_sequences):
                final_hidden_states, _ = self.lstm.forward_propagation(sequence)

                self.lstm.back_propagation_through_time(target_sequence)

                lstm_features.append(final_hidden_states)

                lstm_loss = cost_func(target_sequence, final_hidden_states, 'mse')
                lstm_epoch_loss += lstm_loss

                inc_for_lstm += 1
                if inc_for_lstm % 100 == 0:
                    print("Sequences processed in LSTM: ", inc_for_lstm, '/', train_sequences_length)


            lstm_features_reshaped = np.array(lstm_features).reshape(self.batch_size, self.sequence_length, self.hidden_lstm_neurons)

            self.dense_nn.x = lstm_features_reshaped
            self.dense_nn.y = train_target_sequences

            hidden1_activated_arr, output_activated_arr = self.dense_nn.forward_propagation()
            self.dense_nn.back_propagation(hidden1_activated_arr, output_activated_arr)

            lstm_avg_loss = lstm_epoch_loss / len(train_sequences)
            lstm_losses.append(lstm_avg_loss)

            display_training_info(epoch, self.epochs, lstm_avg_loss, self.dense_nn.mse_values[epoch])


        display_training_time(training_start_time)

        self.lstm.save_model(lstm_losses[-1], self.investment_symbol)
        self.dense_nn.save_model(self.dense_nn.mse_values[-1], self.investment_symbol)

        plot_losses(lstm_losses, 'MSE', 'LSTM')
        plot_losses(self.dense_nn.mse_values, 'MSE', 'Dense')

        return lstm_losses


    def test(self, sequences, target_sequences, lstm_model_info, dense_model_info):
        lstm, dense_nn = self.prepare_models_for_test_or_predict(sequences, lstm_model_info, dense_model_info, False)

        lstm_features = []
        lstm_loss = 0

        for sequence, target_sequence in zip(sequences, target_sequences):
            final_hidden_states, _ = lstm.forward_propagation(sequence)
            lstm.back_propagation_through_time(target_sequence)

            lstm_features.append(final_hidden_states)

            lstm_loss += cost_func(target_sequence, final_hidden_states, 'mse')

        lstm_features_reshaped = np.array(lstm_features).reshape(sequences.shape[0], sequences.shape[1], lstm.hidden_neurons)

        dense_nn.x = lstm_features_reshaped
        dense_nn.y = target_sequences

        hidden1_activated_arr, output_activated_arr = dense_nn.forward_propagation()
        dense_nn.back_propagation(hidden1_activated_arr, output_activated_arr)

        display_training_info(0, 1, lstm_loss / len(target_sequences), dense_nn.mse_values[0])


    def predict_for_n_days(self, sequence, lstm_model_info, dense_model_info, n_days):
        lstm, dense_nn = self.prepare_models_for_test_or_predict(sequence, lstm_model_info, dense_model_info, True)

        lstm_features = []

        final_hidden_states, _ = lstm.forward_propagation(sequence)
        lstm_features.append(final_hidden_states)

        lstm_features_reshaped = np.array(lstm_features).reshape(self.batch_size, sequence.shape[0], lstm.hidden_neurons)

        dense_nn.x = lstm_features_reshaped
        _, output_activated_arr = dense_nn.forward_propagation()

        print(output_activated_arr.shape)
        denorm_output_activated_arr = min_max_denormalization(output_activated_arr, dense_nn.target_sequences_min, dense_nn.target_sequences_max)

        return denorm_output_activated_arr[:, :n_days, :]


    def prepare_models_for_test_or_predict(self, sequences, lstm_model_info, dense_model_info, is_predict):
        lstm = LSTM(
            input_sequence_length=sequences.shape[1] if is_predict else sequences.shape[2],
            hidden_neurons=lstm_model_info["hidden_neurons"],
        )

        lstm.forget_gate_weights = lstm_model_info["forget_gate_weights"]
        lstm.input_gate_weights = lstm_model_info["input_gate_weights"]
        lstm.candidate_gate_weights = lstm_model_info["candidate_gate_weights"]
        lstm.output_gate_weights = lstm_model_info["output_gate_weights"]

        lstm.forget_gate_bias = lstm_model_info["forget_gate_bias"]
        lstm.input_gate_bias = lstm_model_info["input_gate_bias"]
        lstm.candidate_gate_bias = lstm_model_info["candidate_gate_bias"]
        lstm.output_gate_bias = lstm_model_info["output_gate_bias"]

        dense_nn = DenseNN(
            x=np.zeros((sequences.shape[0], sequences.shape[1], self.hidden_lstm_neurons)),
            y=np.zeros((sequences.shape[0], sequences.shape[1], 1)),
            target_sequences_min=dense_model_info["target_sequences_min"],
            target_sequences_max=dense_model_info["target_sequences_max"],
            hidden1_neurons=dense_model_info["hidden1_neurons"],
        )

        dense_nn.input_to_hidden1_weights = dense_model_info["input_to_hidden1_weights"]
        dense_nn.hidden1_to_output_weights = dense_model_info["hidden1_to_output_weights"]

        dense_nn.hidden1_bias = dense_model_info["hidden1_bias"]
        dense_nn.output_bias = dense_model_info["output_bias"]

        return lstm, dense_nn


    def load_models(self, lstm_file_name, dense_file_name):
        lstm_model_info = np.load(f"{PRICE_PREDICTION_SAVED_MODELS_PATH}{self.investment_symbol}/{lstm_file_name}", allow_pickle=True).item()
        dense_model_info = np.load(f"{PRICE_PREDICTION_SAVED_MODELS_PATH}{self.investment_symbol}/{dense_file_name}", allow_pickle=True).item()

        return lstm_model_info, dense_model_info


def display_training_info(epoch, epochs, lstm_avg_loss, dense_nn_loss):
    print("=" * 41)
    print("Epoch  " + str(epoch + 1) + " / " + str(epochs))
    print("LSTM Loss: " + "{:.6f}".format(lstm_avg_loss))
    print("Dense Loss: " + "{:.6f}".format(dense_nn_loss))
    print("=" * 41)

