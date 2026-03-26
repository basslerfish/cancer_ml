"""
Models.
"""

import keras
from keras import layers
import tensorflow as tf



def get_simple_cnn(input_shape: tuple, filter_sizes: list) -> keras.Model:
    """
    Simple CNN model with 3D conv filters.
    """
    input = keras.Input(shape=input_shape)
    x = input

    # downscale
    for fs in filter_sizes:
        x = layers.Conv3D(fs, kernel_size=3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv3D(fs, kernel_size=3, activation="relu", padding="same")(x)

    # upscale
    for fs in filter_sizes[::-1]:
        x = layers.Conv3DTranspose(fs, kernel_size=3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv3DTranspose(fs, kernel_size=3, activation="relu", padding="same")(x)

    output = layers.Conv3D(1, 3, activation="sigmoid", padding="same")(x)
    model = keras.Model(input, output)
    return model


