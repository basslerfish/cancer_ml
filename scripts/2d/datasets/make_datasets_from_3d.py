"""
Quick way to get a 2D dataset: take 3D dataset, split by section.
We also want to remove sections without segmentations.

Right now, we lose information which z-section each sample is, and to which overall sample (=patient) it belongs to.
TODO: retain z-info
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from cancer_ml.preprocess import remove_unannotated_sections

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/3d/samples500_val15_test15_128-128-64")

# load 3d
dsets = {}
for name in ["train", "test", "val"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    dsets[name] = ds

# funcs
def interleave_func(X, y) -> tf.data.Dataset:
    """
    Flatten the z-sections.
    """
    X_shape = X.shape
    X, y, _ = tf.py_function(
        func=remove_unannotated_sections,
        inp=[X, y],
        Tout=[tf.float16, tf.bool, tf.int32]
    )
    new_shape = [None, *X_shape[1:]]  # the first dim changes but the size does not matter since we interleave that dim anyway
    X.set_shape(new_shape)
    y.set_shape(new_shape)
    interleaved_ds = tf.data.Dataset.from_tensor_slices((X, y))
    return interleaved_ds


# do it
output = DSET_FOLDER / "flattened"
for name, ds in dsets.items():
    new_ds = ds.interleave(
        interleave_func,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE,
    )
    path = output / name
    new_ds.save(str(path))
