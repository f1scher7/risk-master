import pandas as pd
import numpy as np
from fischerAI.utils.fischerAI_utils import load_data_csv
from fischerAI.utils.input_data_normalization import *
from enums import Columns


COLUMNS_FOR_TRAINING = [Columns.OPEN.value, Columns.HIGH.value, Columns.LOW.value, Columns.CLOSE.value, Columns.VOLUME.value]


def prepare_training_data(file_name: str):
    data = load_data_csv(file_name, True)

    data[Columns.TIMESTAMP.value] = pd.to_datetime(data[Columns.TIMESTAMP.value], unit="s")

    data.set_index(data[Columns.TIMESTAMP.value], inplace=True)
    data.dropna(inplace=True)

    # print("Data before normalization:")
    # print(data.tail(100))

    data[COLUMNS_FOR_TRAINING] = min_max_normalization(data[COLUMNS_FOR_TRAINING])

    return data

