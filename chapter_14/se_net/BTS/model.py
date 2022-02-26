from unicodedata import name

import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     concatenate, 
                                     GaussianNoise, GlobalAveragePooling2D,
                                     MaxPool2D, Permute, PReLU, Reshape,
                                     UpSampling2D, add, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from loss import custom_loss

K.set_image_data_format("channels_last")
import tensorflow as tf
import tf_slim as slim
from tf_slim.layers import conv2d,  flatten, fully_connected, max_pool2d


def unet_model(input_shape, modified_unet = True, learning_rate = 0.01, start_channel = 64, 
               number_of_levels = 3, inc_rate = 2, output_channels = 4, saved_model_dir = None):
    
    input_layer = Input(shape = input_shape, name = "input_layer")
    with tf.name_scope('unet'):
        with slim.arg_scope([conv2d], padding='same'):
            if modified_unet:
                x = GaussianNoise(0.01, name='GaussianNoise')(input_layer)
                x = conv2d((x),  num_outputs =64, kernel_size=2)
                x = level_block_modified('modified_lvl_blk', x, start_channel, number_of_levels, inc_rate)
                x = BatchNormalization(axis=-1)(x)
                x = PReLU(shared_axes=[1, 2])(x)
            else:
                x = level_block('lvl_blk', x, input_layer, start_channel,number_of_levels, inc_rate)

        x = conv2d((x), output_channels, 1)
        output_layer = Activation('softmax')(x)
        
        model = Model(input_layer, output_layer)
        
    if modified_unet:
        print("Using modified UNet")
    else:
        print("Using standard UNet")
    
    if saved_model_dir:
        model.load_weights(saved_model_dir)
        print(f"Loaded weights from {saved_model_dir}")
    
    sgd = SGD(lr = learning_rate, momentum=0.9, decay=0.0)
    model.compile(optimizer=sgd, loss=custom_loss)
    
    return model


def se_block(name, x, ratio = 16):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = x.shape[channel_axis]
    se_shape = (1, 1, filters)
    with tf.name_scope(name):
        with slim.arg_scope([fully_connected], weights_initializer ='he_normal'):
            se = GlobalAveragePooling2D()(x)
            se = Reshape(se_shape)(se)
            se = fully_connected((se), num_outputs=filters // ratio, activation_fn=tf.nn.relu)
            se = fully_connected((se), num_outputs=filters, activation_fn=tf.nn.sigmoid)
            
            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)
            x = multiply([x, se])
    return x

    

def level_block(name, x, dim, level, inc):
    
    if level > 0:
        m = conv_layers(f'{name}_{level}_conv_layer_m', x, dim)

        x = MaxPool2D(pool_size=(2, 2))(m)
        x = level_block(f'{name}_{level - 1}_res1.0', x, int(dim * inc), level - 1, inc)
        x = UpSampling2D(size=(2, 2))(x)
        x = conv2d((x), filters=dim, kernel_siz=2, padding='same')

        m = concatenate([x, m])
        x = conv_layers(f'{name}_{level}_conv_layer_x', m, dim)
    else:
        x = conv_layers(f'{name}_{level}_conv_layer_x', x, dim)
    return x


@slim.add_arg_scope
def level_block_modified(name, x, dim, level, inc):
    with tf.name_scope(name):
        with slim.arg_scope([conv2d], kernel_size=2,  padding='same'):
            if level > 0:
                m = res_block(f'{level}_res1.0', x, dim, encoder_path=True)
                x = conv2d(m, num_outputs=int(inc * dim), stride=2)
                x = level_block_modified(f'{level - 1}', x, int(dim * inc), level - 1, inc)

                x = UpSampling2D(size=(2, 2))(x)
                x = conv2d(x,num_outputs=dim)

                m = concatenate([x, m])
                m = se_block(f'{level}_seblock2.0', m, 8)
                x = res_block(f'{level}_res2.0', m, dim, encoder_path=False)
            else:
                x = res_block(f'{level}_res1.0', x, dim, encoder_path=True)
            return x   
      
def conv_layers(name, x, dim):
    with tf.name_scope(name):
        with slim.arg_scope([conv2d], num_outputs=dim, kernel_size=3,  padding='same'):
            with slim.arg_scope([Activation], Activation='relu'):
                x = conv2d(x)
                x = Activation()(x)

                x = conv2d(x)
                x = Activation()(x)
                return x


def res_block(name, x, dim, encoder_path = True):
    with tf.name_scope(name):
        with slim.arg_scope([conv2d], num_outputs=dim, kernel_size=3, padding='same'):
            with slim.arg_scope([PReLU], shared_axes=[1,2]):
                with slim.arg_scope([BatchNormalization], axis= -1):
                    m = BatchNormalization()(x)
                    m = PReLU()(m)
                    m = conv2d((m))

                    m = BatchNormalization()(x)
                    m = PReLU()(m)
                    m = conv2d((m))

                    if not encoder_path:
                        x = conv2d(inputs = x,  kernel_size=1)
                    x = add([x, m])
                    return x
    