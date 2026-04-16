"""
Misc functions related to model training.
"""
import keras
import numpy as np

from keras.src.utils.summary_utils import count_params


def get_param_count(model: keras.Model) -> dict:
    """Get number of trainable parameters in a model."""
    trainable = count_params(model.trainable_weights)
    not_trainable = count_params(model.non_trainable_weights)
    param_info = {
        "trainable_weights": trainable,
        "non_trainable_weights": not_trainable,
    }
    return param_info


def get_data_info(dsets: dict) -> dict:
    """
    Get the shape of a batch of training data.
    """
    X, _ = next(iter(dsets["train"].take(1)))
    X = X.numpy()
    data_info = {
        "batch_shape": list(X.shape),
        "min_val": float(np.min(X)),
        "max_val": float(np.max(X)),
    }
    return data_info