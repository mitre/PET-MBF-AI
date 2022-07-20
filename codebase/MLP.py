import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import TensorBoard


def make_model(input_shape,
               num_classes,
               num_outputs,
               hidden_sizes,
               learning_rate,
               drop_prob,
               reg,
               X_train=None):
    """
    This function returns a compiled keras fully connected neural network
    according to the architecture and hyperparameter settings specified in the
    arguments.

    input_shape : int
        Shape of the input data 

    num_classes : int
        Number of classes in the classification problem

    num_outputs : int
        Number of outputs. If 1, then binary of multiclass problem. If 
        num_outputs > 1, then multilabel classification problem.

    hidden_sizes : list of ints
        hidden_sizes[i] represents the number of hidden nodes at layer (i + 1)
        of the NN. len(hidden_sizes) represents the depth of the network.
        i.e. hidden_sizes=[128, 128, 64] specifies a 3 layer network with
        first and second layers having hidden size 128 and third layer of 64
        hidden nodes.

    learning_rate : float
        learning_rate

    drop_prob : float
         Dropout probability -- fraction of connections to drop at training

    reg : float
        L2 regularization strength for each fully connected layer

    X_train : numpy array
        Optional. If included, model will include normalization preprocessing layer
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Include preprocessng layer for normalization if X_train is not None
    if X_train is not None:
        normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalization_layer.adapt(X_train)

        x = normalization_layer(x) 
    
    prev_units = input_shape
    for i, units_i in enumerate(hidden_sizes):
        
        init_stddev = np.sqrt(2.0/prev_units)
        x = Dense(
                units_i,
                input_dim=prev_units,
                kernel_initializer=initializers.RandomNormal(stddev=init_stddev),
                kernel_regularizer=regularizers.l2(reg),
            )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop_prob)(x)
        
        prev_units = units_i
    

    if num_outputs < 2:
        # binary or multiclass
        if num_classes == 2:
            # binary
            activation = "sigmoid"
            units = 1
            metrics=['accuracy', AUC(), Precision(), Recall()]
            loss='binary_crossentropy'
        else:
            # multiclass
            activation = "softmax"
            units = num_classes
            loss = tf.keras.losses.SparseCategoricalCrossentropy() 
            raise NotImplementedError('multiclass metrics not selected')
    else:
        # multilabel
        activation = 'sigmoid'
        units = num_outputs        
        metrics=['accuracy', AUC(multi_label=True)]
        loss='binary_crossentropy'
 
    outputs = Dense(units, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    
    return model
