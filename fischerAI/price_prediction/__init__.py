# import numpy as np
# import pandas as pd
# from data_processing import prepare_data_set
# from fischerAI.price_prediction.data_processing import get_sequences, get_min_max_for_sequences_and_target_sequences_from_saved_models
# from fischerAI.utils.input_data_normalization import min_max_denormalization
# from price_prediction_AI import PricePredictionAI
# from env_loader import BITCOIN_DATA_SET, GOLD_DATA_SET, ETH_DATA_SET, SILVER_DATA_SET
# from enums import InvestmentSymbol
#
#
# if __name__ == "__main__":
#
#     investment_symbol = InvestmentSymbol.SILVER
#     data_set = SILVER_DATA_SET
#
#     hidden_lstm_layers = [128, 128, 128]
#     hidden_dense_layers = [64, 64]
#     epochs = 500
#     learning_rate_lstm = 0.0001
#     learning_rate_dense = 0.0001
#     decay_rate_lstm = 0.999
#
#     lstm_model_file_name = "multi_layer_lstm_model_20250108-235552_best.npy"
#     dense_nn_model_file_name = "dense_nn_model_20250108-235552_best.npy"
#
#     # TRAINING
#     training_data = prepare_data_set(data_set, True)
#     train_sequences, train_sequences_min, train_sequences_max, train_target_sequences, train_target_min, train_target_max = get_sequences(training_data, sequence_length=20)
#
#     print(train_sequences.shape[0])
#
#     price_prediction_ai = PricePredictionAI(investment_symbol=investment_symbol, batch_size=train_sequences.shape[0], sequence_length=train_sequences.shape[1], input_sequence_length=train_sequences.shape[2],
#                                             hidden_lstm_layers=hidden_lstm_layers, hidden_dense_layers=hidden_dense_layers, epochs=epochs, learning_rate_lstm=learning_rate_lstm, learning_rate_dense=learning_rate_dense,
#                                             decay_rate_lstm=decay_rate_lstm, sequences_min=train_sequences_min, sequences_max=train_sequences_max, target_sequences_min=train_target_min, target_sequences_max=train_target_max)
#
#     price_prediction_ai.build_model()
#     price_prediction_ai.train(train_sequences, train_target_sequences)


#     # TEST
#     # test_data = prepare_data_set(BITCOIN_DATA, False)
#     # train_sequences_min, train_sequences_max, train_target_min, train_target_max = get_min_max_for_sequences_and_target_sequences_from_saved_models(lstm_model_file_name, dense_nn_model_file_name, InvestmentSymbol.BITCOIN.value)
#     # test_sequences, test_target_sequences = get_sequences(test_data, sequence_length=90, sequences_min_param=train_sequences_min, sequences_max_param=train_sequences_max, target_sequences_min_param=train_target_min, target_sequences_max_param=train_target_max, is_test=True)
#     #
#
#     # price_prediction_ai = PricePredictionAI(investment_symbol=investment_symbol, batch_size=train_sequences.shape[0],
#     #                                         sequence_length=train_sequences.shape[1],
#     #                                         input_sequence_length=train_sequences.shape[2],
#     #                                         hidden_lstm_layers=hidden_lstm_layers,
#     #                                         hidden_dense_layers=hidden_dense_layers, epochs=epochs,
#     #                                         learning_rate_lstm=learning_rate_lstm,
#     #                                         learning_rate_dense=learning_rate_dense,
#     #                                         sequences_min=train_sequences_min, sequences_max=train_sequences_max,
#     #                                         target_sequences_min=train_target_min,
#     #                                         target_sequences_max=train_target_max)
#
#     # lstm_model_info, dense_model_info = price_prediction_ai.load_models(lstm_model_file_name, dense_nn_model_file_name)
#     # price_prediction_ai.test(test_sequences, test_target_sequences, lstm_model_info, dense_model_info)
#
#     # PREDICT
#     data = prepare_data_set(BITCOIN_DATA_SET, False)
#
#     sequences_min, sequences_max, target_min, target_max = get_min_max_for_sequences_and_target_sequences_from_saved_models(lstm_model_file_name, dense_nn_model_file_name, investment_symbol)
#
#     sequence = get_sequences(data, sequence_length=20, sequences_min_param=sequences_min, sequences_max_param=sequences_max, is_prediction=True)
#
#     price_prediction_ai = PricePredictionAI(investment_symbol=investment_symbol, batch_size=sequence.shape[0],
#                                             sequence_length=sequence.shape[1],
#                                             input_sequence_length=sequence.shape[2],
#                                             hidden_lstm_layers=hidden_lstm_layers,
#                                             hidden_dense_layers=hidden_dense_layers, epochs=epochs,
#                                             learning_rate_lstm=learning_rate_lstm,
#                                             learning_rate_dense=learning_rate_dense,
#                                             sequences_min=sequences_min, sequences_max=sequences_max,
#                                             target_sequences_min=target_min,
#                                             target_sequences_max=target_max)
#
#     lstm_model_info, dense_model_info = price_prediction_ai.load_models(lstm_model_file_name, dense_nn_model_file_name)
#     predictions = price_prediction_ai.predict_for_n_days(sequence, lstm_model_info, dense_model_info, 20)
#
#     print(predictions)
#
#     # # plot_predictions(predictions, InvestmentSymbol.BITCOIN.value, 30)
