import numpy as np
import pandas as pd


# We're using Min-Max normalization:
# when we have small spread in training data (age(18, 80));
# when we want to have a range for training data;
def min_max_normalization(data: pd.DataFrame):
    if (data.max() - data.min()).any() == 0:
        print("Warning! Min-Max normalization cause division by zero")
        return data
    return (data - data.min()) / (data.max() - data.min())


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