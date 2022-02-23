import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tf_slim import arg_scope

class BasicResidualSEBlock(layers.Layer):
    
    expansion = 1
    def __init__(self, in_channels, out_channels, stride, r = 16):
        super(BasicResidualSEBlock, self).__init__()
        
        with arg_scope([layers.Conv2D], padding = 'SAME', kernel_size = 3):
            self.residual = Sequential([
                                        layers.Conv2D(out_channels,  strides=stride, use_bias=False, kernel_initializer='he_normal'),
                                        layers.BatchNormalization(),
                                        layers.ReLU(),
                                        layers.Conv2D(out_channels * self.expansion, 
            
        