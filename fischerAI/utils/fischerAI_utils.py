import pandas as pd
import matplotlib.pyplot as plt
from env_loader import DATA_TRAINING_PATH


def load_data_csv(file_name: str, is_training_data: bool, split_ratio: float = 0.8):
    data = pd.read_csv(f"{DATA_TRAINING_PATH}{file_name}")

    split_index = int(len(data) * split_ratio)

    return data[:split_index] if is_training_data else data[split_index:]


def plot_losses(loss_values, cost_func_name):
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(cost_func_name)

    plt.plot(loss_values, label=cost_func_name, color='blue')
    plt.title(cost_func_name)
    plt.ylabel(cost_func_name)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()