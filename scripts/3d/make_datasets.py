"""
Make datasets split by train, val and test.
Nicely preprocessed for Conv3d use.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from cancer_ml.preprocess import split_sample_folders, load_tf, resize_stacks, clip_t1, zscore_t1, change_dims_3d

# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets")
VAL_FRAC = 0.15
TEST_FRAC = 0.15
X_PX = 128
Y_PX = 128
Z_SECTIONS = 64

# make splits
splits, table = split_sample_folders(SOURCE, VAL_FRAC, TEST_FRAC)
n_total = np.sum([len(x) for x in splits.values()])

# make dataset
dsets = {}
for name, folders in splits.items():
    folders = [str(x) for x in folders]
    ds = tf.data.Dataset.from_tensor_slices(folders)
    dsets[name] = ds

# specify preprocessing
img_shape = (X_PX, Y_PX, Z_SECTIONS)
target_shape = (Z_SECTIONS, X_PX, Y_PX, 1)


def preprocess_func(sample_folder: str) -> tuple:
    """Preprocessing pipeline for 3d data."""
    t1_imgs, gtv_imgs = load_tf(sample_folder)
    t1_imgs = t1_imgs.astype(np.float32)
    gtv_imgs = gtv_imgs.astype(np.float32)
    t1_imgs, gtv_imgs = resize_stacks(t1_imgs, gtv_imgs, img_shape)
    t1_imgs, gtv_imgs = clip_t1(t1_imgs, gtv_imgs)
    t1_imgs, gtv_imgs = zscore_t1(t1_imgs, gtv_imgs)
    t1_imgs, gtv_imgs = change_dims_3d(t1_imgs, gtv_imgs, target_shape)
    t1_imgs = tf.cast(t1_imgs, tf.float16)
    gtv_imgs = tf.cast(gtv_imgs, tf.bool)
    return t1_imgs, gtv_imgs


def map_func(sample_folder: str) -> tuple:
    """tf wrapper around preprocessing."""
    t1_data, gtv_data = tf.py_function(
        func=preprocess_func,
        inp=[sample_folder],
        Tout=[tf.float16, tf.bool]
    )
    t1_data.set_shape(target_shape)  # apparently tf.py_function forgets (?!) the shape (!?)
    gtv_data.set_shape(target_shape)
    return t1_data, gtv_data


# prep output
base_name = f"samples{n_total}_val{VAL_FRAC * 100:.0f}_test{TEST_FRAC * 100:.0f}_{X_PX}-{Y_PX}-{Z_SECTIONS}"
base_folder = OUTPUT / base_name
print(f"Base folder: {base_folder}")
if base_folder.is_dir():
    shutil.rmtree(base_folder)
    print(f"{base_folder} already existed - removed.")
os.makedirs(base_folder)
table.to_csv(base_folder / "sample_ids.csv")

# save dsets
for name in list(dsets.keys()):
    ds = dsets[name]
    ds = ds.map(map_func,  num_parallel_calls=tf.data.AUTOTUNE)
    save_path = base_folder / name
    ds.save(str(save_path))
    print(f"{save_path} saved.")