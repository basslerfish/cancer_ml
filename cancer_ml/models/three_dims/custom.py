"""
Models for 3D data..
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.src.layers import Conv3D
from tensorflow.python.keras.layers import UpSampling3D


def get_simple_cnn(
        input_shape: tuple | list | np.ndarray,
        filter_sizes: list,
        kernel_size: int = 3,
        strides: int = 2,
) -> keras.Model:
    """
    Simple CNN 3d segmentation model.
    Upscaling and downscaling parts.
    """
    input = keras.Input(shape=input_shape)
    x = input

    # downscale
    for fs in filter_sizes:
        x = layers.Conv3D(fs, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
        x = layers.Conv3D(fs, kernel_size=kernel_size, activation="relu", padding="same")(x)

    # upscale
    for fs in filter_sizes[::-1]:
        x = layers.Conv3DTranspose(fs, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
        x = layers.Conv3DTranspose(fs, kernel_size=kernel_size, activation="relu", padding="same")(x)

    output = layers.Conv3D(1, kernel_size, activation="sigmoid", padding="same")(x)
    model = keras.Model(input, output)
    return model



def get_advanced_cnn(
        input_shape: tuple | list | np.ndarray,
        filter_sizes: list,
        kernel_size: int = 3,
        strides: int = 2,
) -> keras.Model:
    """
    We add:
    - residual connections.
    - batch normalization.
    - skip connections.
    - upsampling instead of deconv
    """
    input = keras.Input(shape=input_shape)
    x = input

    # downscale
    for fs in filter_sizes:
        x = residual_strided_block(x, fs, kernel_size, strides)

    # upscale
    for fs in filter_sizes[::-1]:
        x = layers.UpSampling3D(fs, kernel_size=kernel_size, strides=strides, activation="relu", padding="same")(x)
        x = layers.UpSampling3D(fs, kernel_size=kernel_size, activation="relu", padding="same")(x)

    output = layers.Conv3D(1, kernel_size, activation="sigmoid", padding="same")(x)
    model = keras.Model(input, output)
    return model


def residual_strided_block(x: tf.Tensor, fs: int, ks: int, strides: int) -> tf.Tensor:
    residual = x

    # process x
    x = layers.Conv3D(fs, ks, padding="same")(x)  # feature extraction
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(fs, ks, strides=strides, padding="same")(x)  # downsampling
    x = layers.BatchNormalization()(x)

    # process residual
    residual = layers.Conv3D(1, 1, strides=strides, padding="same")(residual)
    residual = layers.BatchNormalization()(residual)

    # add
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    return x


def upsample_block(x) -> tf.Tensor:
    x = UpSampling3D()(x)
    x = UpSampling3D()(x)
    x = Conv3D()(x)
    x = Conv3D()(x)