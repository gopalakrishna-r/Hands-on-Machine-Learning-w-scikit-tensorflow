import re
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layer
from functools import partial


class InceptionStem(keras.models.Model):
    def __init__(self, name = 'inception_v1'):
        super(InceptionStem, self).__init__(name=name)
        self.conv2d_1 = Conv2d(name = "conv2d_1", filters = 64, kernel_size=(7,7), strides=2)
        self.maxpool_1 = MaxPool(name="maxpool_1", pool_size=(3, 3), strides=2)
        self.batchNorm_1 = layer.BatchNormalization(name="batchNorm_1")
        self.conv2d_2 = Conv2d(name="conv2d_2", filters=64,kernel_size=(1, 1), strides=1)
        self.conv2d_3 = Conv2d(name="conv2d_3", filters=192,kernel_size=(3, 3), strides=1)
        self.maxpool_2 = MaxPool(name="maxpool_2", pool_size=(3, 3), strides=2)
        
        self.inception_blk_1 = InceptionModule(module_name = "mixed_inc_blk_1", kernels_layer_1_1x1_conv_1=96, kernels_layer_1_1x1_conv_2=16,
                                               kernels_layer_2_1x1_Conv_3=32, kernels_layer_2_3x3_conv_1=128, kernels_layer_2_5x5_conv_2=32, 
                                               kernels_layer_3_1x1_conv_1= 64)
        self.inception_blk_2 = InceptionModule(module_name = "mixed_inc_blk_2", kernels_layer_1_1x1_conv_1=128, kernels_layer_1_1x1_conv_2=32,
                                               kernels_layer_2_1x1_Conv_3=64, kernels_layer_2_3x3_conv_1=192, kernels_layer_2_5x5_conv_2=96, 
                                               kernels_layer_3_1x1_conv_1= 128)
        
        self.maxpool_3 = MaxPool(name="maxpool_3",  pool_size=(3, 3), strides=2)
        
        self.inception_blk_3 = InceptionModule(module_name = "mixed_inc_blk_3", kernels_layer_1_1x1_conv_1=96, kernels_layer_1_1x1_conv_2=16,
                                               kernels_layer_2_1x1_Conv_3=64, kernels_layer_2_3x3_conv_1=208, kernels_layer_2_5x5_conv_2=48, 
                                               kernels_layer_3_1x1_conv_1= 192)
        self.inception_blk_4 = InceptionModule(module_name = "mixed_inc_blk_4", kernels_layer_1_1x1_conv_1=112, kernels_layer_1_1x1_conv_2=24,
                                               kernels_layer_2_1x1_Conv_3=64, kernels_layer_2_3x3_conv_1=224, kernels_layer_2_5x5_conv_2=64, 
                                               kernels_layer_3_1x1_conv_1= 160)
        self.inception_blk_5 = InceptionModule(module_name = "mixed_inc_blk_5", kernels_layer_1_1x1_conv_1=128, kernels_layer_1_1x1_conv_2=24,
                                               kernels_layer_2_1x1_Conv_3=64, kernels_layer_2_3x3_conv_1=256, kernels_layer_2_5x5_conv_2=64, 
                                               kernels_layer_3_1x1_conv_1= 128)
        self.inception_blk_6 = InceptionModule(module_name = "mixed_inc_blk_6", kernels_layer_1_1x1_conv_1=144, kernels_layer_1_1x1_conv_2=32,
                                               kernels_layer_2_1x1_Conv_3=64, kernels_layer_2_3x3_conv_1=288, kernels_layer_2_5x5_conv_2=64, 
                                               kernels_layer_3_1x1_conv_1= 112)
        self.inception_blk_7 = InceptionModule(module_name="mixed_inc_blk_7", kernels_layer_1_1x1_conv_1=160, kernels_layer_1_1x1_conv_2=32,
                                               kernels_layer_2_1x1_Conv_3=128, kernels_layer_2_3x3_conv_1=320, kernels_layer_2_5x5_conv_2=128, 
                                               kernels_layer_3_1x1_conv_1= 256)
        
        self.maxpool_4 = MaxPool(name="maxpool_4", pool_size=(3, 3), strides=2)
        
        self.inception_blk_8 = InceptionModule(module_name="mixed_inc_blk_8", kernels_layer_1_1x1_conv_1=160, kernels_layer_1_1x1_conv_2=32,
                                               kernels_layer_2_1x1_Conv_3=128, kernels_layer_2_3x3_conv_1=320, kernels_layer_2_5x5_conv_2=128,
                                               kernels_layer_3_1x1_conv_1= 256)
        self.inception_blk_9 = InceptionModule(module_name = "mixed_inc_blk_9", kernels_layer_1_1x1_conv_1=192, kernels_layer_1_1x1_conv_2=48,
                                               kernels_layer_2_1x1_Conv_3=128, kernels_layer_2_3x3_conv_1=384, kernels_layer_2_5x5_conv_2=128,
                                               kernels_layer_3_1x1_conv_1= 384)
        self.flatten = layer.Flatten(name="flatten")
        self.globalAvgPool = layer.AveragePooling2D()
        self.dropout = layer.Dropout(rate=0.4)
        self.fcn = layer.Dense(name = "FCN", units=1000, activation=tf.nn.relu)
        self.softmax = layer.Dense(
            name="softmax", units=10, activation=tf.nn.softmax)
    
    def build(self, input_shape):
        super(InceptionStem, self).build(input_shape)
    
    def call(self, x):
        x = self.conv2d_1(x)
        x = self.maxpool_1(x)
        x = self.batchNorm_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.maxpool_2(x)
        x = self.inception_blk_1(x)
        x = self.inception_blk_2(x)
        x = self.maxpool_3(x)
        x = self.inception_blk_3(x)
        x = self.inception_blk_4(x)
        x = self.inception_blk_5(x)
        x = self.inception_blk_6(x)
        x = self.inception_blk_7(x)
        x = self.maxpool_4(x)
        x = self.inception_blk_8(x)
        x = self.inception_blk_9(x)
        x = self.globalAvgPool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fcn(x)
        x = self.softmax(x)
        return x
        

Conv2d = partial(keras.layers.Conv2D, kernel_size=(1, 1), strides=1, padding='SAME', activation = tf.nn.relu)
MaxPool = partial(keras.layers.MaxPool2D, pool_size=(3, 3),  strides=1, padding='SAME')

    
class InceptionModule(keras.layers.Layer):

    def __init__(self, module_name,
                 kernels_layer_1_1x1_conv_1, kernels_layer_1_1x1_conv_2,
                 kernels_layer_2_3x3_conv_1, kernels_layer_2_5x5_conv_2, kernels_layer_2_1x1_Conv_3,
                 kernels_layer_3_1x1_conv_1
                 ):
        super().__init__(name=module_name)
        self.layer_1_1x1_conv_1 = Conv2d(
            name="layer_1_1x1_conv_1", filters=kernels_layer_1_1x1_conv_1)
        self.layer_1_1x1_conv_2 = Conv2d(
            name="layer_1_1x1_conv_2", filters=kernels_layer_1_1x1_conv_2)
        self.layer_1_3x3_MaxPool = MaxPool(name="layer_1_3x3_MaxPool")
        self.layer_2_3x3_conv_1 = Conv2d(
            name="layer_2_3x3_conv_1", filters=kernels_layer_2_3x3_conv_1, kernel_size=(3, 3))
        self.layer_2_5x5_conv_2 = Conv2d(
            name="layer_2_5x5_conv_2", filters=kernels_layer_2_5x5_conv_2, kernel_size=(3, 3))
        self.layer_2_1x1_Conv_3 = Conv2d(
            name="layer_2_1x1_Conv_3", filters=kernels_layer_2_1x1_Conv_3)
        self.layer_3_1x1_conv_1 = Conv2d(
            name="layer_3_1x1_conv_1", filters=kernels_layer_3_1x1_conv_1)

    def call(self, x):
        depth_1_output = self.layer_3_1x1_conv_1(x)

        depth_2_output = self.layer_1_1x1_conv_1(x)
        depth_2_output = self.layer_2_3x3_conv_1(depth_2_output)

        depth_3_output = self.layer_1_1x1_conv_2(x)
        depth_3_output = self.layer_2_5x5_conv_2(depth_3_output)

        depth_4_output = self.layer_1_3x3_MaxPool(x)
        depth_4_output = self.layer_2_1x1_Conv_3(depth_4_output)

        return tf.concat([depth_1_output, depth_2_output, depth_3_output, depth_4_output], axis=3)


