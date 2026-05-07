"""
Very simple CNN.
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
    """
    Very simple encoder-decoder CNN for 2D segmentation.
    Two conv layers per stage, first one with strides.
    No skip connections, no residuals, no norming.
    """
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    assert len(input_shape) == 3
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

