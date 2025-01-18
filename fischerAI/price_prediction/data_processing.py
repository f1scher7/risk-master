import numba
import pandas as pd
import numpy as np
from fischerAI.utils.fischerAI_utils import load_data_csv
from fischerAI.utils.input_data_normalization import *
from enums import Column
from env_loader import PRICE_PREDICTION_SAVED_MODELS_PATH
from enums import InvestmentSymbol

COLUMNS_FOR_TRAINING = [Column.CLOSE.value]


def prepare_data_set(file_name: str, is_training_data: bool):
    data = load_data_csv(file_name, is_training_data, split_ratio=0)

    if InvestmentSymbol.BITCOIN.value in file_name:
        data[Column.TIMESTAMP.value] = pd.to_datetime(data[Column.TIMESTAMP.value], unit="s")

        data.set_index(data[Column.TIMESTAMP.value], inplace=True)
        data.dropna(inplace=True)

        data['date'] = data.index.date

        daily_data = data.groupby('date').apply(lambda x: x.iloc[[-1]])
    else:
        if InvestmentSymbol.GOLD.value in file_name:
            data['Date'] = pd.to_datetime(data['Date'], format="%b %d, %Y")
        elif InvestmentSymbol.SILVER.value in file_name or InvestmentSymbol.ETHEREUM.value in file_name:
            data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
            data = data.iloc[::-1]

        data.set_index('Date', inplace=True)
        data.dropna(inplace=True)

        data['date'] = data.index.date

        daily_data = data.groupby('date').apply(lambda x: x.iloc[[-1]])
        daily_data.reset_index(drop=True, inplace=True)

    return daily_data


def  get_sequences(data, sequence_length, sequences_min_param=None, sequences_max_param=None, target_sequences_min_param=None, target_sequences_max_param=None, is_prediction=False, is_test=False, is_graph=False):
    data_np = data[COLUMNS_FOR_TRAINING].values
    targets_np = data[Column.CLOSE.value].values

    if is_graph:
        return data_np[-sequence_length:].flatten() # convert to 1 dim

    if is_prediction:
        last_sequence_norm = min_max_normalization_with_min_max_params(data_np[-sequence_length:], sequences_min_param, sequences_max_param)
        return last_sequence_norm.reshape(1, last_sequence_norm.shape[0], last_sequence_norm.shape[1])

    sequences, target_sequences =  create_sequences_numba(data_np, targets_np.reshape(-1, 1), sequence_length)

    if is_test:
        sequences_norm = min_max_normalization_with_min_max_params(sequences, sequences_min_param, sequences_max_param)
        target_sequences_norm = min_max_normalization_with_min_max_params(target_sequences, target_sequences_min_param, target_sequences_max_param)
        return sequences_norm, target_sequences_norm

    sequences_norm, sequences_min, sequences_max, = min_max_normalization(sequences)
    target_sequences_norm, target_sequences_min, target_sequences_max, = min_max_normalization(target_sequences)

    return sequences_norm, sequences_min, sequences_max, target_sequences_norm, target_sequences_min, target_sequences_max


def get_min_max_for_sequences_and_target_sequences_from_saved_models(lstm_file_name, dense_file_name, investment_symbol):
    if lstm_file_name is None and dense_file_name is None:
        if investment_symbol == 'gold':
            lstm_file_name = 'multi_layer_lstm_model_20250115-192715.npy'
            dense_file_name = 'dense_nn_model_20250115-192715.npy'
        elif investment_symbol == 'btc':
            lstm_file_name = 'multi_layer_lstm_model_20250108-235552_best.npy'
            dense_file_name = 'dense_nn_model_20250108-235552_best.npy'
        elif investment_symbol == 'silver':
            lstm_file_name = 'multi_layer_lstm_model_20250116-111026.npy'
            dense_file_name = 'dense_nn_model_20250116-111026.npy'
        elif investment_symbol == 'eth':
            lstm_file_name = 'multi_layer_lstm_model_20250115-233430.npy'
            dense_file_name = 'dense_nn_model_20250115-233430.npy'

    lstm_model_info = np.load(f"{PRICE_PREDICTION_SAVED_MODELS_PATH}{investment_symbol}/{lstm_file_name}",allow_pickle=True).item()
    dense_model_info = np.load(f"{PRICE_PREDICTION_SAVED_MODELS_PATH}{investment_symbol}/{dense_file_name}", allow_pickle=True).item()

    return lstm_model_info, dense_model_info, lstm_model_info["sequences_min"], lstm_model_info["sequences_max"], dense_model_info["target_sequences_min"], dense_model_info["target_sequences_max"]


@numba.jit(nopython=True, parallel=True)
def create_sequences_numba(data_np, targets_np, sequence_length):
    num_sequences = len(data_np) - 2 * sequence_length + 1
    num_features = data_np.shape[1]

    sequences = np.zeros((num_sequences, sequence_length, num_features))
    target_sequences = np.zeros((num_sequences, sequence_length, 1))

    for i in numba.prange(num_sequences):  # we are using prange for parallelization
        sequences[i] = data_np[i:i + sequence_length]
        target_sequences[i] = targets_np[i + sequence_length:i + sequence_length * 2]

    return sequences, target_sequences