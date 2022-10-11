from functools import partial

from tensorflow.keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, concatenate
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from unet_met.net import create_convolution_block
from unet_met.net import create_convolution_block_one
from unet_met.net import create_convolution_block_two
from unet_met.metrics import dice_coefficient_loss

import tensorflow as tf
import numpy as np
#from keras.utils import multi_gpu_model
from tensorflow import keras
keras.backend.set_image_data_format('channels_first')

#create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU)


## sigmoid, relu
def cao2019_model(input_shape=(1, 64, 64, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=1, n_labels=2, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)
        
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))
        
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)
        
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
        print(current_layer.shape)
    #mm=current_layer[0,:,:,:,:]
    ## first layer, flatten
    #mid_1=create_convolution_block(current_layer, 4096, kernel=(3, 3, 3),strides=(2, 2, 2))
    #mid_1_dropout=SpatialDropout3D(0.3)(mid_1)

    
    mid_f1=Flatten()(current_layer)
    
    
    mid_2=Dense(512,activation='sigmoid')(mid_f1)
    mid_2=Dropout(0.5)(mid_2)
    
    mid_3=Dense(512,activation='sigmoid')(mid_2)
    mid_3=Dropout(0.5)(mid_3)
    
    mid_4=Dense(2, activation='sigmoid')(mid_3)
    mid_4=Dropout(0.5)(mid_4)
    
    final_convolution=mid_4
    activation_block = Activation(activation_name)(final_convolution)
    #print(activation_block)


    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    #strategy = tf.distribute.MirroredStrategy(gpus)
    #with strategy.scope():
    if gpus:
        for gpu in gpus:
            with tf.device(gpu.name):
                model = Model(inputs=inputs, outputs=activation_block)
                model.compile(loss="categorical_crossentropy", optimizer=optimizer(lr=initial_learning_rate))
        #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    #pa_model=multi_gpu_model(model, gpus=4)
    #pa_model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    #pa_model.compile(optimizer=optimizer(lr=initial_learning_rate), loss='categorical_crossentropy')
    return model




def create_context_module(input_layer, n_level_filters, dropout_rate=0.5, data_format="channels_first"):
    convolution1 = create_convolution_block_one(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block_one(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



