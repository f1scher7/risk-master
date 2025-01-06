import numba
import pandas as pd
import numpy as np
from fischerAI.utils.fischerAI_utils import load_data_csv
from fischerAI.utils.input_data_normalization import *
from enums import Column


COLUMNS_FOR_TRAINING = [Column.OPEN.value, Column.HIGH.value, Column.LOW.value, Column.CLOSE.value, Column.VOLUME.value]


def prepare_data_set(file_name: str, is_training_data: bool):
    data = load_data_csv(file_name, is_training_data)

    data[Column.TIMESTAMP.value] = pd.to_datetime(data[Column.TIMESTAMP.value], unit="s")

    data.set_index(data[Column.TIMESTAMP.value], inplace=True)
    data.dropna(inplace=True)

    # print("Data before normalization:")
    # print(data.tail(100))

    data['date'] = data.index.date

    daily_data = data.groupby('date').apply(lambda x: x.iloc[[-1]])

    # print(len(daily_data))

    daily_data[COLUMNS_FOR_TRAINING], data_min_denorm, data_max_denorm = min_max_normalization(daily_data[COLUMNS_FOR_TRAINING])

    return daily_data, data_min_denorm, data_max_denorm


def get_sequences(data, sequence_length, is_prediction=False):
    data_np = data[COLUMNS_FOR_TRAINING].values
    targets_np = data[Column.CLOSE.value].values

    if is_prediction:
        last_sequence_norm, _, _ = min_max_normalization(data_np[-sequence_length:])
        return last_sequence_norm

    sequences, target_sequences =  create_sequences_numba(data_np, targets_np.reshape(-1, 1), sequence_length)

    sequences_norm, _, _, = min_max_normalization(sequences)
    target_sequences_norm, target_sequences_min, target_sequences_max, = min_max_normalization(target_sequences)

    return sequences_norm, target_sequences_norm, target_sequences_min, target_sequences_max


@numba.jit(nopython=True, parallel=True)
def create_sequences_numba(data_np, targets_np, sequence_length):
    num_sequences = len(data_np) - sequence_length - sequence_length + 1
    num_features = data_np.shape[1]

    sequences = np.zeros((num_sequences, sequence_length, num_features))
    target_sequences = np.zeros((num_sequences, sequence_length, 1))

    for i in numba.prange(num_sequences):  # we are using prange for parallelization
        sequences[i] = data_np[i:i + sequence_length]
        target_sequences[i] = targets_np[i + sequence_length:i + sequence_length * 2]

    return sequences, target_sequences