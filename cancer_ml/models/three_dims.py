"""
Models for 3D data..
"""

import keras
import numpy as np
from keras import layers


class DiceBCELoss(keras.losses.Loss):
    """
    Combined Dice and BCE loss.
    BCE loss on segmentation is not very good predictor of performance when class imbalance is strong.
    Dice is a better fit, but may be so bad at beginning that we supplement with BCE.
    """
    def __init__(self, dice_weight: float = 0.5, **kwargs) -> None:
        super().__init__(name="DiceBCELoss", **kwargs)
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight

    def call(self, y_true, y_pred):
        dice_loss = keras.losses.Dice()(y_true, y_pred)
        bce_loss = keras.losses.BinaryCrossentropy()(y_true, y_pred)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return combined_loss




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


