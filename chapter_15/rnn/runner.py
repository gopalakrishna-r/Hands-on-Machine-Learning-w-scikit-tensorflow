from chapter_15.generator import generate_time_series
from model import build_model, build_rnn_model, build_drnn_model, build_drnn_dense_model, build_lrn_model
import tensorflow.keras as keras
import numpy as np
from chapter_15.evaluator import evaluate
import argparse

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_model', help='comma seperated list of models to use.', default='1')
    args = parser.parse_args()
    if args.rnn_model:
        rnn_model = int(args.rnn_model)
        if rnn_model == 1:
            net = build_model()
            print(f'mse for Sequential model {evaluate(net, X_test, y_test, X_valid, y_valid, X_train, y_train)}')
        elif rnn_model == 2:
            rnn_net = build_rnn_model()
            print(f'mse for rnn model {evaluate(rnn_net, X_test, y_test, X_valid, y_valid, X_train, y_train)}')
        elif rnn_model == 3:
            drnn_net = build_drnn_model()
            print(f'mse for drnn model {evaluate(drnn_net, X_test, y_test, X_valid, y_valid, X_train, y_train)}')
        elif rnn_model == 4:
            drnn_dense_net = build_drnn_dense_model()
            print(f'mse for drnn dense model '
                  f'{evaluate(drnn_dense_net, X_test, y_test, X_valid, y_valid, X_train, y_train)}')

        else:
            y_pred = X_valid[:, -1]
            print(f'mse for no model {np.mean(keras.losses.mean_squared_error(y_valid, y_pred))}')
