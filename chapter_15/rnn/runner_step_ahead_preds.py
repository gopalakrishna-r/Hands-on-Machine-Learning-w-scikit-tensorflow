import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from chapter_15.evaluator import evaluate
from chapter_15.rnn.normalization.LNSimpleRNNCell import LNSimpleRNNCell
from chapter_15.rnn.runner import y_valid
from chapter_15.visualizer import plot_multiple_forecasts
from chapter_15.generator import generate_time_series, generate_datasets
from model import build_drnn_dense_step_ahead_model, build_drnn_dense_model, build_lrn_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_model', help='comma seperated list of models to use.', default='1')
    args = parser.parse_args()
    n_steps = 50
    if args.rnn_model:
        rnn_model = int(args.rnn_model)
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        loss = keras.losses.mean_squared_error
        if rnn_model == 1:
            np.random.seed(42)
            tf.random.set_seed(42)

            X_train, Y_train, X_valid, Y_valid, X_test, Y_test = generate_datasets(n_steps)
            drnn_net = build_drnn_dense_model()
            evaluate(drnn_net, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

            np.random.seed(43)
            series = generate_time_series(1, n_steps + 10)
            X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
            X = X_new
            for step_ahead in range(10):
                Y_pred_one = drnn_net.predict(X[:, step_ahead:])[:, np.newaxis, :]
                X = np.concatenate([X, Y_pred_one], axis=1)
            Y_pred = X[:, n_steps:]

            plot_multiple_forecasts(X_new, Y_new, Y_pred)

            np.random.seed(42)
            series = generate_time_series(10000, n_steps + 10)
            X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
            X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
            X_test, Y_test = series[9000:, :n_steps], series[9000, -10:, 0]

            drnn_net = build_drnn_dense_step_ahead_model()

            drnn_net.compile(loss='mse', optimizer='adam')
            history = drnn_net.fit(X_train, Y_train,
                                   validation_data=(X_valid, y_valid), epochs=20, workers=8)

            np.random.seed(43)
            series = generate_time_series(1, 50 + 10)
            X_new, Y_new = series[:, :50, :], series[:, -10:, :]
            Y_pred = drnn_net.predict(X_new)[..., np.newaxis]

            plot_multiple_forecasts(X_new, Y_new, Y_pred)
        if rnn_model == 2:
            np.random.seed(42)
            series = generate_time_series(10000, n_steps + 10)
            X_train = series[:7000, :n_steps]
            X_valid = series[7000:9000, :n_steps]
            X_test = series[9000:, :n_steps]
            Y = np.empty((10000, n_steps, 10))
            for step_ahead in range(1, 10 + 1):
                Y[..., step_ahead - 1] = series[..., step_ahead: step_ahead + n_steps, 0]
            Y_train = Y[:7000]
            Y_valid = Y[7000:9000]
            Y_test = Y[9000:]

            print(X_train.shape, Y_train.shape)

            np.random.seed(42)
            tf.random.set_seed(42)

            model = keras.models.Sequential([
                keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                                 input_shape=[None, 1]),
                keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
                keras.layers.TimeDistributed(keras.layers.Dense(10))
            ])

            def last_time_step_mse(y_true, y_pred):
                return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])

            model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
            history = model.fit(X_train, Y_train, epochs=20,
                                validation_data=(X_valid, Y_valid))
