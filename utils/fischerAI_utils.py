import pandas as pd
from env_loader import DATA_TRAINING_PATH


def load_data_csv(file_name: str, is_training_data: bool, split_ratio: float = 0.8):
    data = pd.read_csv(f"{DATA_TRAINING_PATH}{file_name}")

    split_index = int(len(data) * split_ratio)

    return data[:split_index] if is_training_data else data[split_index:]
