"""
Make a tensorflow dataset.
There are a lot of decisions here already.
- resizing the scans so that they have the same shape
- rescaling them

Finally, keras expects the shape (batch_size, n_images, x, y, n_channels)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf

from src.load import find_files, load_images

# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
N_SAMPLES = 100
TARGET_SHAPE = (128, 256, 256, 1)
NAME = "train"
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets")

# get paths
folders = list(SOURCE.iterdir())
folders = list(filter(lambda x: x.is_dir(), folders))
folders_to_load = np.random.choice(folders, size=N_SAMPLES, replace=False)
df = pd.DataFrame({"i": np.arange(folders_to_load.size), "sample": [x.name for x in folders_to_load]})
df.to_csv(OUTPUT / f"{NAME}_{N_SAMPLES}.csv")
t1_paths = []
gtv_paths = []
for folder in folders_to_load:
    t1_file, gtv_file = find_files(folder)
    t1_paths.append(str(t1_file))
    gtv_paths.append(str(gtv_file))

# load images
path_ds = tf.data.Dataset.from_tensor_slices((t1_paths, gtv_paths))
zoom_shape = (TARGET_SHAPE[1], TARGET_SHAPE[2], TARGET_SHAPE[0])


def load_and_preprocess(t1_tf, gtv_tf) -> tuple:
    """Load and resize"""
    # convert to str
    t1_str = t1_tf.numpy().decode("utf-8")
    gtv_str = gtv_tf.numpy().decode("utf-8")
    assert isinstance(t1_str, str), type(t1_str)

    # load shape: x, y, n_images
    t1_data, gtv_data = load_images(t1_str, gtv_str)
    assert t1_data.ndim == 3

    # set data type
    t1_data = t1_data.astype(np.float32)
    gtv_data = gtv_data.astype(np.float32)

    #  rescale
    current_shape = t1_data.shape
    zoom_factors = [t / c for t, c in zip(zoom_shape, current_shape)]
    t1_data = scipy.ndimage.zoom(t1_data, zoom=zoom_factors, order=1)
    gtv_data = scipy.ndimage.zoom(gtv_data, zoom=zoom_factors, order=0)

    # clip
    low, high = np.percentile(t1_data, [0.5, 99.5])
    t1_data = np.clip(t1_data, low, high)

    # zscore
    t1_data = (t1_data - np.mean(t1_data)) / (np.std(t1_data) + 10 ** -8)

    # shift axis and add empty channel axis
    t1_data = tf.transpose(t1_data, [2, 0, 1])
    gtv_data = tf.transpose(gtv_data, [2, 0, 1])
    t1_data = tf.expand_dims(t1_data, axis=-1)
    gtv_data = tf.expand_dims(gtv_data, axis=-1)
    assert np.all(t1_data.shape == TARGET_SHAPE), t1_data.shape
    assert np.all(gtv_data.shape == TARGET_SHAPE), gtv_data.shape

    # change gtv data to save space
    t1_data = tf.cast(t1_data, tf.float32)
    gtv_data = tf.cast(gtv_data, tf.bool)
    return t1_data, gtv_data


def map_func(t1_tf, gtv_tf) -> tuple:
    """Wrapper around load_and_process."""
    t1_data, gtv_data = tf.py_function(
        func=load_and_preprocess,
        inp=[t1_tf, gtv_tf],
        Tout=[tf.float32, tf.bool]
    )
    t1_data.set_shape(TARGET_SHAPE)  # apparently tf.py_function forgets (?!) the shape (!?)
    gtv_data.set_shape(TARGET_SHAPE)
    return t1_data, gtv_data


# save
img_ds = path_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
save_dir = OUTPUT / f"{NAME}_{N_SAMPLES}"
if save_dir.is_dir():
    shutil.rmtree(save_dir)
    print("Previous dset deleted.")
img_ds.save(str(save_dir))
print(f"{save_dir} saved")
