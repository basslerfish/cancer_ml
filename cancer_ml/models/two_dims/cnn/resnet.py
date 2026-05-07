"""
ResNet encoder decoder with optional U-net style skip connections.
TODO: the actual ResNet had repeated stages
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers


def residual_strided_block(
        x: tf.Tensor,
        filter_size: int,
        kernel_size: int,
        strides: int,
        dropout_rate: float,
) -> tf.Tensor:
    """
    A downsampling block with 2x conv2d, batch norm and residuals.
    1st conv has kernel size of 3, second has kernel size of 1.
    This block design follows the 2015 "Deep Residual Learning for Image Recognition" paper.
    ReLU is replaced with Swish.
    Dropout is optional, apparently BatchNorm and dropout don't always get along.
    """
    residual = x
    x = layers.Conv2D(filter_size, kernel_size, padding="same", strides=strides, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    x = layers.Conv2D(filter_size, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Conv2D(filter_size, 1, strides=strides, padding="same", use_bias=False)(residual)
    residual = layers.BatchNormalization()(residual)

    x = layers.Add()([x, residual])
    x = layers.Activation("swish")(x)
    return x


def upsampling_block(
        x: tf.Tensor,
        filter_size: int,
        kernel_size: int,
        strides: int,
) -> tf.Tensor:
    """Upsampling block for ResNet CNN. Mirrors downsampling block."""
    # upsample (same for x and residual)
    x = layers.UpSampling2D(size=strides)(x)
    residual = x

    # convolutions
    x = layers.Conv2D(filter_size, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filter_size, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Conv2D(filter_size, 1, padding="same", use_bias=False)(residual)
    residual = layers.BatchNormalization()(residual)

    x = layers.Add()([x, residual])
    x = layers.Activation("swish")(x)
    return x


def get_resnet_cnn(
        input_shape: list | tuple | np.ndarray,
        filter_sizes: list | tuple | np.ndarray,
        kernel_size: int = 3,
        strides: int = 2,
        add_skips: bool = False,
        dropout_rate: float = 0,
) -> keras.Model:
    """
    Encoder-decoder CNN with ResNet like blocks and optional skip connections.
    """
    if len(input_shape) == 4:  # batch_size, x, y, n_channels
        input_shape = input_shape[1:]
    assert len(input_shape) == 3
    input = keras.Input(input_shape)

    # encoder
    x = input
    if add_skips:
        skips = []
    for fs in filter_sizes:
        if add_skips:
            skips.append(x)
        x = residual_strided_block(x, fs, strides=strides, kernel_size=kernel_size, dropout_rate=dropout_rate)

    # decoder
    for i_fs, fs in enumerate(filter_sizes[::-1]):
        x = upsampling_block(x, fs, strides=strides, kernel_size=kernel_size)
        if add_skips:
            this_skip = skips[::-1][i_fs]
            x = layers.Concatenate()([x, this_skip])

    output = layers.Conv2D(1, kernel_size, activation="sigmoid", padding="same")(x)
    model = keras.Model(input, output)
    return model


def get_flexible_model(
        input_shape: list | tuple | np.ndarray,
        filter_sizes: list | tuple | np.ndarray,
        model_type: str,
        kernel_size: int = 3,
        strides: int = 2,
        add_skips: bool = False,
) -> keras.Model:
    """For hyperparameter tuning."""
    if model_type == "simple":
        model = get_simple_cnn(
            input_shape=input_shape, filter_sizes=filter_sizes,
            strides=strides, kernel_size=kernel_size,
        )
    elif model_type == "advanced":
        model = get_advanced_cnn(
            input_shape=input_shape, filter_sizes=filter_sizes,
            strides=strides, kernel_size=kernel_size, add_skips=add_skips,
        )
    else:
        raise ValueError(f"{model_type=} unknown")
    return model