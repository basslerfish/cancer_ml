"""
Misc utility functions.
"""

import argparse
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


def get_args_dirs(also_tb: bool = True) -> tuple:
    """
    On the cluster, we may receive data_dir, output_dir and tb_dir as inputs to the script.
    """
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    if also_tb:
        parser.add_argument("--tb_dir")
    args = parser.parse_args()

    # check paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    assert data_dir.is_dir(), f"{data_dir} does not exist"
    assert output_dir.is_dir(), f"{output_dir} does not exist"
    if also_tb:
        tb_dir = Path(args.tb_dir)
        assert tb_dir.is_dir(), f"{tb_dir} does not exist"
        return data_dir, output_dir, tb_dir
    else:
        return data_dir, output_dir