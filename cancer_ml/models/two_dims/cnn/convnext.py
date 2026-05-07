"""
Implement the ConvNeXt architecture.

Here are the changes:
- Separate downscaling
- LayerNormalization instead of BatchNormalization
"""
import keras
from keras import layers
import numpy as np


def convnext_block(
        x: keras.KerasTensor,
) -> keras.KerasTensor:
    """
    3 conv layers and a residual.
    1st conv layer is depthwise separable and has large kernel size.
    Other two conv layers are 1x1.
    Does not downscale (no strides).
    """
    assert x.ndim == 4
    fs = x.shape[1]
    bottleneck_dim = fs * 4  # actually inverse bottleneck
    residual = x
    x = layers.DepthwiseConv2D(fs, kernel_size=7)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(bottleneck_dim, kernel_size=1)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Conv2D(fs, kernel_size=1)(x)
    x = x + residual
    return x


def downsample_block(x: keras.KerasTensor) -> keras.KerasTensor:
    """
    Downsampling occurs in a separate block at the beginning of each stage.
    2x2 kernel with strides 2
    """
    assert x.ndim == 4
    fs = x.shape[1]
    x = layers.Conv2D(fs, strides=2)(x)
    x = layers.LayerNormalization()(x)
    return x


def upsample_block(x):
    return x


def make_model(
        input_shape: list | tuple | np.ndarray,
        stage_repeats: list = (1, 1, 3, 1),
        use_skip: bool = True,
) -> keras.Model:
    """
    ConvNeXt-based CNN.
    :param input_shape:
    :param stage_repeats:
    :return:
    """
    input = keras.Input(input_shape)
    x = input

    # downsample
    if use_skip:
        skip_vals = []
    for i_stage, n_repeats in enumerate(stage_repeats):
        x = downsample_block(x)
        for i_repeat in range(n_repeats):
            x = convnext_block(x)

    # upsample
    for i_stage in range(len(stage_repeats)):
        x = upsample_block(x)
        if use_skip:
            pass

    output = x
    model = keras.Model(input, output)
    return model