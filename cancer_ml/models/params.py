import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def get_data_params(X: tf.Tensor) -> dict:
    X = X.numpy()
    hparams = {
        "batch_shape": list(X.shape),
        "min_val": float(np.min(X)),
        "max_val": float(np.max(X)),
    }
    return hparams