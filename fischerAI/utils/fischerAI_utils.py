import pandas as pd
import matplotlib.pyplot as plt
from env_loader import DATA_TRAINING_PATH


def load_data_csv(file_name: str, is_training_data: bool, split_ratio: float = 0.8):
    data = pd.read_csv(f"{DATA_TRAINING_PATH}{file_name}")

    split_index = int(len(data) * split_ratio)

    return data[:split_index] if is_training_data else data[split_index:]


def plot_mse(mse_values):
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title('MSE')

    plt.plot(mse_values, label='MSE', color='blue')
    plt.title('Mean Squared Error over Epochs')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()