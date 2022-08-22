import tensorflow.keras as keras
from normalization.LNSimpleRNNCell import LNSimpleRNNCell


def build_model():
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=[50, 1]),
        keras.layers.Dense(1)
    ])


def build_rnn_model():
    return keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ])


def build_drnn_model():
    return keras.models.Sequential([
        keras.layers.SimpleRNN(20, input_shape=[None, 1], return_sequences=True),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1)])


def build_drnn_dense_model():
    return keras.models.Sequential([
        keras.layers.SimpleRNN(20, input_shape=[None, 1], return_sequences=True),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)])


def build_drnn_dense_step_ahead_model():
    return keras.models.Sequential([
        keras.layers.SimpleRNN(20, input_shape=[None, 1], return_sequences=True),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(10)
    ])


def build_sequence_sequence_model():
    return keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])


def build_lrn_model():
    return keras.models.Sequential([
        keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True, input_shape=[None, 1]),
        keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
