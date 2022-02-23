import tensorflow as tf
import tensorflow.keras as keras
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3,
                        strides=1, padding="same", use_bias=False)

class ResidualNet(keras.layers.Layer):
    def __init__(self, filters, strides=1 , activation=tf.nn.relu, **kwargs):
        super().__init__( **kwargs)
        self.activation = activation
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(), 
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters , kernel_size = 1, strides = strides), 
                keras.layers.BatchNormalization()
            ]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
            
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
            
        return self.activation(Z + skip_Z)
