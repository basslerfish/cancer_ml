"""
Misc functions related to model training.
"""
from pathlib import Path

import keras
import numpy as np
import yaml
from keras.src.utils.summary_utils import count_params


def get_recursive_description(model: keras.Model | keras.Layer, depth: int = 0) -> None:
    """Get recursive description of a model's layers."""
    for layer in model.layers:
        trainable = layer.trainable
        print(depth, "\t" * depth, f"{layer.name} (trainable={trainable}, params={layer.count_params():,})")
        if hasattr(layer, "layers"):
            get_recursive_description(layer, depth + 1)


def get_param_count(model: keras.Model, verbose: bool = True) -> dict:
    """Get number of trainable parameters in a model."""
    trainable = count_params(model.trainable_weights)
    not_trainable = count_params(model.non_trainable_weights)
    param_info = {
        "trainable_weights": trainable,
        "non_trainable_weights": not_trainable,
    }
    if verbose:
        print("Model params:")
        for k, v in param_info.items():
            print(f"\t {k} -> {v:,}")
    return param_info


def get_data_info(dsets: dict, verbose: bool = True) -> dict:
    """
    Get the shape of a batch of training data.
    """
    n_batches = dsets["train"].cardinality()
    X, _ = next(iter(dsets["train"].take(1)))
    X = X.numpy()
    n_samples = n_batches * X.shape[0]
    data_info = {
        "n_batches": n_batches,
        "est_total_samples": n_samples,
        "batch_shape": list(X.shape),
        "min_val": float(np.min(X)),
        "max_val": float(np.max(X)),
    }
    if verbose:
        print("Data info:")
        for k, v in data_info.items():
            print(f"\t {k} -> {v}")
    return data_info


def load_config_yaml(config_file: Path, verbose: bool = True) -> dict:
    """Load a yaml file."""
    with open(config_file) as file:
        config = yaml.safe_load(file)
    if verbose:
        print("Model config: ")
        for k, v in config.items():
            if isinstance(v, dict):
                print(f"\t {k}:")
                for sub_k, sub_v in v.items():
                    print(f"\t\t {sub_k} -> {sub_v}")
            else:
                print(f"\t {k} -> {v}")
    return config