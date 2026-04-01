
"""
Make 2D datasets from scratch.
"""
from pathlib import Path

import tensorflow as tf

from cancer_ml.preprocess import (load_tf, resize_stacks, clip_t1, change_dims_3d, zscore_t1, minmax_t1,
                                  remove_unannotated_sections, split_sample_folders)


# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d")
RESIZE_SHAPE = [256, 256, 128]
KERAS_SHAPE = [128, 256, 256, 1]
SCALING = "minmax"
VAL_FRAC = TEST_FRAC = 0.15
MIN_PX = 20

# filter files
splits, df = split_sample_folders(SOURCE, VAL_FRAC, TEST_FRAC)
n_samples = df.shape[0]

#
def load_and_preprocess(sample_path) -> tuple:
    """Get out nice 3D stacks."""
    X, y = load_tf(sample_path)
    X, y = resize_stacks(X, y, RESIZE_SHAPE)
    X, y = clip_t1(X, y)
    if SCALING == "minmax":
        X, y = minmax_t1(X, y)
    if SCALING == "zscore":
        X, y = zscore_t1(X, y)
    X, y = change_dims_3d(X, y, KERAS_SHAPE)
    X, y, _ = remove_unannotated_sections(X, y, MIN_PX)
    assert X.ndim == 4
    assert y.ndim == 4
    return X, y


# funcs
def interleave_func(sample_path) -> tf.data.Dataset:
    """
    Flatten the z-sections.
    """
    X, y = tf.py_function(
        func=load_and_preprocess,
        inp=[sample_path],
        Tout=[tf.float16, tf.bool]
    )
    new_shape = [None, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1]
    X.set_shape(new_shape)
    y.set_shape(new_shape)
    interleaved_ds = tf.data.Dataset.from_tensor_slices((X, y))
    return interleaved_ds


# go!
dset_name = f"samples{n_samples}_{SCALING}_val{VAL_FRAC*100:.0f}_test{TEST_FRAC*100:.0f}_{RESIZE_SHAPE[0]}-{RESIZE_SHAPE[1]}"
print(dset_name)
dset_folder = OUTPUT / dset_name
for name, folders in splits.items():
    print(f"---Split: {name}---")
    ds = tf.data.Dataset.from_tensor_slices([str(x) for x in folders])
    ds = ds.interleave(interleave_func, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=tf.data.AUTOTUNE)
    save_path = dset_folder / name
    ds.save(str(save_path))