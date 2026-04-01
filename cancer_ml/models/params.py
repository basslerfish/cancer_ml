import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def write_hparams(hparams: dict, file_path: Path) -> None:
    """Write hyperparams dict"""
    with open(file_path, "w") as f:
        json.dump(hparams, f)


def read_hparams(file_path: Path) -> dict:
    """Read hyperparams file."""
    with open(file_path, "r") as f:
        hparams = json.load(f)
    return hparams


def get_data_params(X: tf.Tensor) -> dict:
    X = X.numpy()
    hparams = {
        "X_shape": list(X.shape),
        "X_min": float(np.min(X)),
        "X_max": float(np.max(X)),
    }
    return hparams