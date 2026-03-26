"""
Make a tensorflow dataset.
There are a lot of decisions here already.
- resizing the scans so that they have the same shape
- rescaling them
"""
from pathlib import Path

import numpy as np
import scipy.ndimage
import tensorflow as tf

from src.load import find_files, load_images

# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
N_SAMPLES = 100
TARGET_SHAPE = (256, 256, 128)
NAME = "train"
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets")

# get paths
folders = list(SOURCE.iterdir())
folders = list(filter(lambda x: x.is_dir(), folders))
folders_to_load = np.random.choice(folders, size=N_SAMPLES)
t1_paths = []
gtv_paths = []
for folder in folders_to_load:
    t1_file, gtv_file = find_files(folder)
    t1_paths.append(str(t1_file))
    gtv_paths.append(str(gtv_file))

# load images
path_ds = tf.data.Dataset.from_tensor_slices((t1_paths, gtv_paths))


def load_and_preprocess(t1_tf, gtv_tf) -> tuple:
    """Load and resize"""
    t1_str = t1_tf.numpy().decode("utf-8")
    gtv_str = gtv_tf.numpy().decode("utf-8")
    assert isinstance(t1_str, str), type(t1_str)
    t1_data, gtv_data = load_images(t1_str, gtv_str)

    # set data type
    t1_data = t1_data.astype(np.float32)
    gtv_data = gtv_data.astype(np.float32)

    #  rescale
    current_shape = t1_data.shape
    zoom_factors = [t / c for t, c in zip(TARGET_SHAPE, current_shape)]
    t1_data = scipy.ndimage.zoom(t1_data, zoom=zoom_factors, order=1)
    gtv_data = scipy.ndimage.zoom(gtv_data, zoom=zoom_factors, order=0)
    assert np.all(t1_data.shape == np.array(TARGET_SHAPE))
    assert np.all(t1_data.shape == np.array(TARGET_SHAPE))

    # clip
    low, high = np.percentile(t1_data, [0.5, 99.5])
    t1_data = np.clip(t1_data, low, high)

    # zscore
    t1_data = (t1_data - np.mean(t1_data)) / (np.std(t1_data) + 10 ** -8)
    return t1_data, gtv_data


def map_func(t1_tf, gtv_tf) -> tuple:
    """Wrapper around load_and_process."""
    t1_data, gtv_data = tf.py_function(
        func=load_and_preprocess,
        inp=[t1_tf, gtv_tf],
        Tout=[tf.float32, tf.float32]
    )
    t1_data.set_shape(TARGET_SHAPE)  # apparently tf.py_function forgets (?!) the shape (!?)
    gtv_data.set_shape(TARGET_SHAPE)
    return t1_data, gtv_data


# save
img_ds = path_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
save_file = str(OUTPUT / f"{NAME}_{N_SAMPLES}")
img_ds.save(save_file)
