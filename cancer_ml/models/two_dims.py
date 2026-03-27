"""
Models for 2D convolution
"""
import keras
import numpy as np
from keras import layers


def get_simple_cnn(
        input_shape: list | tuple | np.ndarray,
        filter_sizes: list | tuple | np.ndarray,
        kernel_size: int = 3,
        strides: int = 2,
) -> keras.Model:
    """Very simple 2D CNN for segmentation."""
    input = keras.Input(input_shape)
    x = input
    for fs in filter_sizes:
        x = layers.Conv2D(fs, kernel_size, strides=strides, activation="relu", padding="same")(x)
        x = layers.Conv2D(fs, kernel_size, activation="relu", padding="same")(x)

    for fs in filter_sizes[::-1]:
        x = layers.Conv2DTranspose(fs, kernel_size, strides=strides, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(fs, kernel_size, activation="relu", padding="same")(x)

    output = layers.Conv2D(1, kernel_size, activation="sigmoid", padding="same")(x)
    model = keras.Model(input, output)
    return model


def get_advanced_cnn() -> None:
    pass