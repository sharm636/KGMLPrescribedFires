# IMPORT LIBRARIES

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import backend as bk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, ConvLSTM3D, Conv3D, Masking, SpatialDropout3D, Bidirectional
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers.core import SpatialDropout2D
from keras import backend as K

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

import numpy as np
import pandas as pd

# CL model
# Base model for PGCL model
x_train = datasets[0]
def cl_model():
    '''
     We construct 4 `ConvLSTM2D` layers,
     followed by a `Conv3D` layer for 
     the spatiotemporal model.
    '''
    inp = layers.Input(shape=(None, *x_train.shape[2:]))
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(10, 10),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    model = keras.models.Model(inp, x)
    return model


# MDN - CL model
# Base model for PGCL+
def cl_mdn_model():
    inp = layers.Input(shape=(None, *x_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(10, 10),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    
    # mixing parameters are p1 and p2
    # Gaussian component mean parameters are m1 and m2
    # Gaussian component st dev parameters are v1 and v2
    m1 = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    v1 = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    m2 = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    v2 = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    p1, p2 = tf.Variable(.5, trainable=False), tf.Variable(.5, trainable=False)
    v1, v2 = tf.math.softplus(v1), tf.math.softplus(v2)
    x = p1* (
        m1+v1*tfd.Normal(0, 1).sample(tf.shape(m1))
    ) +p2 * (
        m1+v2*tfd.Normal(0, 1).sample(tf.shape(m2))
    ) 

    model = keras.models.Model(inp, x)
    return model

