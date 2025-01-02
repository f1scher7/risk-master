import numba
import pandas as pd
import numpy as np
from fischerAI.utils.fischerAI_utils import load_data_csv
from fischerAI.utils.input_data_normalization import *
from enums import Column


COLUMNS_FOR_TRAINING = [Column.OPEN.value, Column.HIGH.value, Column.LOW.value, Column.CLOSE.value, Column.VOLUME.value]


def prepare_training_data(file_name: str):
    data = load_data_csv(file_name, True)

    data[Column.TIMESTAMP.value] = pd.to_datetime(data[Column.TIMESTAMP.value], unit="s")

    data.set_index(data[Column.TIMESTAMP.value], inplace=True)
    data.dropna(inplace=True)

    # print("Data before normalization:")
    # print(data.tail(100))

    data['date'] = data.index.date

    daily_data = data.groupby('date').apply(lambda x: x.iloc[[0, len(x) // 2, -1]])

    daily_data[COLUMNS_FOR_TRAINING] = min_max_normalization(daily_data[COLUMNS_FOR_TRAINING])

    return daily_data


def get_sequences(data, sequence_length):
    data_np = data[COLUMNS_FOR_TRAINING].values
    targets_np = data[Column.CLOSE.value].values

    return create_sequences_numba(data_np, targets_np, sequence_length)


@numba.jit(nopython=True, parallel=True)
def create_sequences_numba(data_np, targets_np, sequence_length):
    num_samples = len(data_np) - sequence_length
    num_features = data_np.shape[1]

    sequences = np.zeros((num_samples, sequence_length, num_features))
    targets = np.zeros(num_samples)

    for i in numba.prange(num_samples):  # we are using prange for parallelization
        sequences[i] = data_np[i:i + sequence_length]
        targets[i] = targets_np[i + sequence_length]

    return sequences, targets