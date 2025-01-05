import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from env_loader import DATA_SETS_PATH
from time import perf_counter


def load_data_csv(file_name: str, is_training_data: bool, split_ratio: float = 0.8):
    data = pd.read_csv(f"{DATA_SETS_PATH}{file_name}")

    data = data[int(len(data) / 2):]

    split_index = int(len(data) * split_ratio)

    return data[:split_index] if is_training_data else data[split_index:]


def save_nn_model(model_info, file_path, investment_symbol, file_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = f"{file_path}/{investment_symbol}/{file_name}_{timestamp}.npy"

    np.save(file, model_info)
    print(f"{file_name} was saved to {file}")


def plot_losses(loss_values, cost_func_name, title):
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(cost_func_name)

    plt.plot(loss_values, label=cost_func_name, color='blue')
    plt.title(title)
    plt.ylabel(cost_func_name)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def display_training_time(training_start_time):
    training_time = perf_counter() - training_start_time

    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f'Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s')
