import keras

def assert_gpu_available():
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
