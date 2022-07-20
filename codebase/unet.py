import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import TensorBoard

def fully_connected(x, num_classes, num_outputs, fc_hidden_sizes, drop_prob, reg):
    
    for units_i in fc_hidden_sizes:
        x = layers.Dense(
                units_i,
                kernel_regularizer=regularizers.l2(reg)
            )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dropout(drop_prob)(x)

    
    if num_outputs < 2:
        # binary or multiclass
        if num_classes == 2:
            # binary
            activation = "sigmoid"
            units = 1
        else:
            # multiclass
            activation = "softmax"
            units = num_classes
            raise NotImplementedError('multiclass metrics not selected')
    else:
        # multilabel
        activation = 'sigmoid'
        units = num_outputs        
 
    x = layers.Dense(units, activation=activation)(x)

    return x

def dense_block(x, f, k, dilated=True):
    steps = np.int((f/2)/k)
    
    for i in range(0,steps):
        n = layers.Conv2D(f//2, 1, padding="same", use_bias=False)(x)
        n = layers.BatchNormalization()(n)
        n = layers.Activation("relu")(n)
        n = layers.Conv2D(k, 3, padding="same", use_bias=False)(n)
        n = layers.BatchNormalization()(n)
        n = layers.Activation("relu")(n)
        x = layers.concatenate([x,n])

        return x
    
    
def make_model(input_shape=[48,48,1],
               num_classes=2,
               num_outputs=1, 
               f=16,
               k=4,
               l=3,
               fc_hidden_sizes=[128],
               learning_rate=None,
               drop_prob=0,
               reg=0):    
    """
    This function returns a fully dense U-Net adapted for classification. See 
    "Fully Dense UNet for 2D Sparse Photoacoustic Tomography Artifact Removal"
    by Guan, et al. (2018).
    
    input_shape : tuple
        Shape of input data.

    num_classes : int
        Number of classes for categorical classification. Ignored if num_ouputs
        > 1.

    num_outputs : int
        Number of outputs. If num_outputs is 1, prediction problem is
        binary or categorical. If num_outputs is > 1, prediction problem is
        multilabel, with num_ouputs binary labels.

    f : int
        Initial number of feature maps.
    
    k : int
        Initial value for growth rate for dense blocks

    l : int
        Number of descending and ascending blocks in the fully dense U-Net

    fc_hidden_sizes : list of ints
        hidden_sizes[i] represents the number of hidden nodes at layer (i + 1)
        of the fully connected layers of the NN appended to the U-Net to adapt
        it to classification rather than segmentation. len(hidden_sizes) 
        represents the number of hidden layers.
        i.e. hidden_sizes=[128, 128, 64] specifies 3 hidden layers with
        first and second hidden layers having size 128 and third hidden layer
        of 64.

    learning_rate : float

    drop_prob : float
        Float between 0 and 1. Fraction of the input units to drop.

    reg : float
        L2 regularization strength for each fully connected layer.


    """
    outputs = {}
    inputs = layers.Input(shape=input_shape)
    saved_maps = [] ## To be concatenated later during upsampling
    
    ## First Layer to learn f//2 features
    ## initial 1 --> 32 
    init_layer = layers.Conv2D(f//2, 3, padding="same", use_bias=False, input_shape=(input_shape[0],input_shape[1],input_shape[2]))(inputs)
    init_layer = layers.BatchNormalization()(init_layer)
    x = layers.Activation("relu")(init_layer)
    
    ## Contracting Path
    for level in range(0,l):
        F = f * 2**(level)
        K = k * 2**(level)
        
        print('Contracting Level:' + str(level) + ' f:' + str(F) + ' k:' + str(K))
        x = dense_block(x, F, K)
        
        if (level < l-1):
            saved_maps.append(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)
    
    ## Expanding Path
    for level in range(0,l-1):
        F = f * 2**(l-level-2)
        K = k * 2**(l-level-2)
        print('Expanding Level:' + str(l-level-2) + ' f:' + str(F) + ' k:' + str(K))
        x = layers.Conv2DTranspose(F, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.concatenate([x,saved_maps[l-level-2]])
        x = layers.Conv2D(F//2, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = dense_block(x, F, K)


    
    x = tf.keras.layers.Flatten()(x)
    x = fully_connected(x, num_classes, num_outputs, fc_hidden_sizes, drop_prob, reg)
    outputs = x

    if num_outputs < 2:
        # binary or multiclass
        if num_classes == 2:
            # binary
            metrics=['accuracy', AUC(), Precision(), Recall()]
            loss='binary_crossentropy'
        else:
            # multiclass
            loss = tf.keras.losses.SparseCategoricalCrossentropy() 
            raise NotImplementedError('multiclass metrics not selected')
    else:
        # multilabel
        metrics=['accuracy', AUC(multi_label=True)]
        loss='binary_crossentropy'

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if learning_rate:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam()
    
    
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    
    return model
