"""
Misc utility functions.
"""
import argparse
import json
from pathlib import Path

import keras


def assert_gpu_available() -> None:
    """Assert that we have access to the GPU."""
    backend = keras.backend.backend()
    print(f"Backend: {backend}")
    if backend == "tensorflow":
        import tensorflow as tf

        # Checks if any GPU is visible to TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) > 0, "No GPU found for TensorFlow backend!"
    elif backend == "torch":
        import torch

        # Specifically checks for CUDA (NVIDIA) support
        assert torch.cuda.is_available(), "CUDA is not available for PyTorch backend!"
    elif backend == "jax":
        import jax

        # Checks if the default backend is GPU
        assert jax.default_backend() == "gpu", "JAX is not using GPU!"
    else:
        raise ValueError(f"{backend=} unknown")


def get_image_size_folder_name(folder: Path) -> tuple:
    """
    Folder names eg:
    samples500_val15_test15_128-128
    samples500_minmax_val15_test15_128-128
    """
    parts = folder.name.split("_")
    image_size = parts[-1]
    image_size = image_size.split("-")
    image_size = [int(x) for x in image_size]
    image_size = tuple(image_size)
    return image_size
