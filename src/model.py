import keras
from keras import layers
import tensorflow as tf

def get_binary_cnn(input_shape: tuple, filter_sizes: list) -> keras.Model:
    """
    CNN that returns for every img in the stack whether it contains cancer or not.
    Note: this is not very useful.
    """
    input = keras.Input(shape=input_shape)
    x = input
    for i_fs, fs in enumerate(filter_sizes):
        x = layers.Conv2D(fs, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.Conv2D(fs, kernel_size=3, activation="relu", padding="same")(x)
        if i_fs == len(filter_sizes) - 1:
            x = layers.GlobalAvgPool2D()(x)
        else:
            x = layers.MaxPooling2D(2, padding="same")(x)
    output = layers.Dense(input_shape[-1], activation="sigmoid")(x)
    model = keras.Model(input, output)
    return model


