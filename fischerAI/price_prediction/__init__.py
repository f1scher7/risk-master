import pandas as pd
from data_processing import prepare_training_data
from env_loader import BITCOIN_DATA


if __name__ == "__main__":
    data = prepare_training_data(BITCOIN_DATA)

    print("Data after normalization:")
    print(data.tail(20))
