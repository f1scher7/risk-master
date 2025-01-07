import numpy as np
import pandas as pd


# We're using Min-Max normalization:
# when we have small spread in training data (age(18, 80));
# when we want to have a range for training data;
def min_max_normalization(data, feature_range=(0, 1)):
    data_min = np.min(data)
    data_max = np.max(data)
    min_range, max_range = feature_range

    if np.any(data_max - data_min == 0):
        print("Warning! Min-Max normalization causes division by zero")
        return data, None, None


    scale = (max_range - min_range) / (data_max - data_min)
    normalized_data = (data - data_min) * scale + min_range

    return normalized_data, data_min, data_max


def min_max_normalization_with_min_max_params(data, data_min, data_max, feature_range=(0, 1)):
    min_range, max_range = feature_range

    scale = (max_range - min_range) / (data_max - data_min)
    normalized_data = (data - data_min) * scale + min_range

    return normalized_data


def min_max_denormalization(data_norm, data_min_denorm, data_max_denorm, feature_range=(0, 1)):
    min_range, max_range = feature_range

    scale = (max_range - min_range) / (data_max_denorm - data_min_denorm)
    denormalized_data = (data_norm - min_range) / scale + data_min_denorm

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