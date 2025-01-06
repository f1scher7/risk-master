import numpy as np
import pandas as pd


# We're using Min-Max normalization:
# when we have small spread in training data (age(18, 80));
# when we want to have a range for training data;
def min_max_normalization(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    if np.any(data_max - data_min == 0):
        print("Warning! Min-Max normalization causes division by zero")
        return data, None, None

    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data, data_min, data_max


def min_max_denormalization(data_norm, data_min_denorm, data_max_denorm):
    denormalized_data = (data_norm * (data_max_denorm - data_min_denorm)) + data_min_denorm

    return denormalized_data


# We're using Log normalization:
# when we have large numbers in training data;
# when we have large spread in training data (price(1000$-1000000$));
def log_normalization(data: pd.DataFrame):
    if (data <= 0).any().any():
        shift_value = abs(data.min().min()) + 1
        data += shift_value

        print("Warning! Data contains <=0 values")
        print(f"Data shifted by {shift_value}")

    return np.log(data + 1)