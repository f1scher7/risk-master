import numpy as np


# We're using Min-Max normalization:
# when we have small spread in training data (age(18, 80));
# when we want to have a range for training data;
def min_max_normalization(x, normalization_range, min_val=None, max_val=None):
    if min_val is None:
        min_val = x.min(axis=0)
    if max_val is None:
        max_val = x.max(axis=0)

    min_range, max_range = normalization_range

    # For normalization range (0, 1)
    normalized_X = (x - min_val) / (max_val - min_val + 1e-10)

    # For normalization range (a, b); for the (0, 1) normalization_X = scaled_X
    scaled_X = normalized_X * (max_range - min_range) + min_range

    return scaled_X, min_val, max_val


# We're using Log normalization:
# when we have large numbers in training data;
# when we have large spread in training data (price(1000$-1000000$));
def log_normalization(x):
    x += 1e-10 # We're adding a small value to avoid log(0)
    return np.log(x)
