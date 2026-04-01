
"""
Make 2D datasets from scratch.
"""
from pathlib import Path

import tensorflow as tf

from cancer_ml.preprocess import (load_tf, resize_stacks, clip_t1, change_dims_3d, zscore_t1, minmax_t1,
                                  remove_unannotated_sections, split_sample_folders, )


# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d")
RESIZE_SHAPE = [512, 512, 64]  # x, y, z
SCALING = "uint8"  # minmax, uint8, zscore
VAL_FRAC = TEST_FRAC = 0.15
MIN_PX = 50  # let's say 10 for 128 x 128, maybe more for higher res

# filter files
splits, df = split_sample_folders(SOURCE, VAL_FRAC, TEST_FRAC)
n_samples = df.shape[0]

# sanity check
assert SCALING in ["uint8", "minmax", "zscore"]
keras_shape = [RESIZE_SHAPE[2], RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1]  # z, x, y, n_channels
if SCALING == "uint8":
    t1_dtype = tf.uint8
else:
    t1_dtype = tf.float16

#
def load_and_preprocess(sample_path) -> tuple:
    """Get out nice 3D stacks."""
    X, y = load_tf(sample_path)
    X, y = resize_stacks(X, y, RESIZE_SHAPE)
    X, y = clip_t1(X, y)
    if SCALING in ["minmax", "uint8"]:
        X, y = minmax_t1(X, y)
        if SCALING == "uint8":
            X = X * 255
    if SCALING == "zscore":
        X, y = zscore_t1(X, y)
    X, y = change_dims_3d(X, y, keras_shape)
    X, y, _ = remove_unannotated_sections(X, y, MIN_PX)
    X = tf.cast(X, t1_dtype)
    y = tf.cast(y, tf.bool)
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
        Tout=[t1_dtype, tf.bool]
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