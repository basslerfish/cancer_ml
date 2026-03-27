"""
Models for 2D convolution
"""
import keras
import numpy as np
from keras import layers


def get_simple_cnn(
        input_shape: list | tuple | np.ndarray,
        filter_sizes: list | tuple | np.ndarray,
) -> keras.Model:
    input = keras.Input(input_shape)

    for fs in filter_sizes:
        x = layers.Conv2D()(x)
        x = layers.Conv2D()(x)

    for fs in filter_sizes[::-1]:
        x = layers.Conv2DTranspose()(x)
        x = layers.Conv2DTranspose()(x)

    output = layers.Conv2D(1, activation="sigmoid")
    model = keras.Model(input, output)
    return model


def get_advanced_cnn() -> None:
    pass