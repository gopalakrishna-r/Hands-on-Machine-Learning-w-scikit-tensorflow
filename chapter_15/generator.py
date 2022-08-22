import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offset2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)

    return series[..., np.newaxis].astype(np.float32)


def generate_datasets( n_steps):
    series = generate_time_series(10000, n_steps + 1)
    x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    x_test, y_test = series[9000:, :n_steps], series[9000:, -1]
    return x_train, y_train, x_valid, y_valid, x_test, y_test
