"""
Quick way to get a 2D dataset: take 3D dataset, split by section.
We also want to remove sections without segmentations.

Right now, we lose information which z-section each sample is, and to which overall sample (=patient) it belongs to.
TODO: retain z-info
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/samples500_val15_test15_128-128-64")
REMOVE_NOSEG = True

# load 3d
dsets = {}
for name in ["train", "test", "val"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    dsets[name] = ds

#
def remove_unannotated_sections(X: tf.Tensor, y: tf.Tensor, n_min: int = 10) -> tuple:
    """
    Remove z-sections without a min amount of cancer.
    """
    X = X.numpy()
    y = y.numpy().astype(np.float32)
    assert X.ndim == 4, f"{X.ndim}"
    n_segmented = np.sum(y, axis=(1, 2, 3))
    is_selected = n_segmented >= n_min
    n_selected = np.sum(is_selected)
    n_total = n_segmented.size
    frac_selected = n_selected / n_total
    print(f"Sections with mask: {n_selected}/{n_total} -> {frac_selected:.1%}")
    X = X[is_selected, :, :, :]
    y = y[is_selected, :, :, :]
    X = tf.convert_to_tensor(X)
    y = tf.convert_to_tensor(y)
    X = tf.cast(X, tf.float16)
    y = tf.cast(y, tf.bool)
    return X, y


def interleave_func(X, y) -> tf.data.Dataset:
    """
    Flatten the z-sections.
    """
    X_shape = X.shape
    y_shape = y.shape
    X, y = tf.py_function(
        func=remove_unannotated_sections,
        inp=[X, y],
        Tout=[tf.float16, tf.bool]
    )
    X.set_shape(X_shape)
    y.set_shape(y_shape)
    interleaved_ds = tf.data.Dataset.from_tensor_slices((X, y))
    return interleaved_ds


# do it
output = DSET_FOLDER.parent / "flattened"
for name, ds in dsets.items():
    new_ds = ds.interleave(
        interleave_func,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE,
    )
    path = output / name
    new_ds.save(str(path))
