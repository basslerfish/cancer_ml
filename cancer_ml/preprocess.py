"""
Preprocessing functions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from cancer_ml.load import find_sample_folders, find_t1_and_gtv_files, load_images


def split_sample_folders(
        source: Path,
        val_frac: float,
        test_frac: float,
        shuffle: bool = True,
        seed: int | None = None,
        limit_samples: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Split data folders into train, val and test set.
    """
    sample_folders = find_sample_folders(source)
    n_samples = len(sample_folders)
    print(f"{n_samples} samples in {source}")
    if limit_samples is not None:
        print(f"Limiting samples to {limit_samples}")
        sample_folders = sample_folders[:limit_samples]

    n_val = int(n_samples * val_frac)
    n_test = int(n_samples * test_frac)
    n_train = n_samples - (n_val + n_test)
    print(f"Val samples: {n_val}")
    print(f"Test samples: {n_test}")
    print(f"Train samples: {n_train}")

    sample_folders = np.array(sample_folders)
    if shuffle:
        rng = np.random.default_rng(seed)
        sample_folders = rng.permutation(sample_folders)

    val_folders = sample_folders[:n_val]
    test_folders = sample_folders[n_val:n_val + n_test]
    train_folders = sample_folders[n_val + n_test:]

    dset_folders = {
        "train": train_folders,
        "val": val_folders,
        "test": test_folders,
    }
    df = []
    for name, folders in dset_folders.items():
        count = 0
        for sample in folders:
            entry = {
                "sample": sample.name,
                "split": name,
                "i_sample_in_ds": count
            }
            df.append(entry)
            count += 1
    df = pd.DataFrame(df)
    return dset_folders, df


def load_tf(sample_folder) -> tuple:
    """Load T1 and GTV images."""
    sample_folder = sample_folder.numpy().decode("utf-8")
    sample_folder = Path(sample_folder)
    assert sample_folder.is_dir()
    print(f"\t Loading {sample_folder.name}.")
    sample_folder = Path(sample_folder)
    assert sample_folder.is_dir()
    t1_file, gtv_file = find_t1_and_gtv_files(sample_folder)
    t1_imgs, gtv_imgs = load_images(t1_file, gtv_file)     # load shape: x, y, n_images
    assert t1_imgs.ndim == 3
    return t1_imgs, gtv_imgs


def resize_stacks(
        t1_imgs: np.ndarray,
        gtv_imgs: np.ndarray,
        target_shape: tuple | list
) -> tuple:
    """
    Resize images to achieve a consistent shape.
    Note: target_shape is X, Y, Z at this point.
    Only after change_dims_3d do we have Z, X, Y, n_channels (as expected by keras for conv)
    """
    current_shape = t1_imgs.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    t1_imgs = scipy.ndimage.zoom(t1_imgs, zoom=zoom_factors, order=1)
    gtv_imgs = scipy.ndimage.zoom(gtv_imgs, zoom=zoom_factors, order=0)
    return t1_imgs, gtv_imgs


def clip_t1(t1_imgs: np.ndarray, gtv_imgs: np.ndarray) -> tuple:
    """Clip T1 range but leave GTV images unaffected."""
    low, high = np.percentile(t1_imgs, [0.5, 99.5])
    t1_data = np.clip(t1_imgs, low, high)
    return t1_data, gtv_imgs


def zscore_t1(t1_imgs: np.ndarray, gtv_imgs: np.ndarray) -> tuple:
    """Z-score T1 values but leave GTV images unaffected."""
    t1_imgs = (t1_imgs - np.mean(t1_imgs)) / (np.std(t1_imgs) + 10 ** -8)
    return t1_imgs, gtv_imgs


def minmax_t1(t1_imgs: np.ndarray, gtv_imgs: np.ndarray) -> tuple:
    """Rescale T1 values to [0, 1] but leave GTB images unaffected."""
    t1_imgs = (t1_imgs - np.min(t1_imgs)) / (np.max(t1_imgs) - np.min(t1_imgs))
    return t1_imgs, gtv_imgs


def change_dims_3d(t1_imgs: np.ndarray, gtv_imgs: np.ndarray, target_shape: list | tuple) -> tuple:
    """
    Reorder axes and add channel axis to prepare for keras processing.
    keras's Conv3D  expects shape: n_imgs, x, y, n_channels
    """
    t1_imgs = tf.transpose(t1_imgs, [2, 0, 1])
    gtv_imgs = tf.transpose(gtv_imgs, [2, 0, 1])
    t1_imgs = tf.expand_dims(t1_imgs, axis=-1)
    gtv_imgs = tf.expand_dims(gtv_imgs, axis=-1)
    assert np.all(t1_imgs.shape == target_shape), t1_imgs.shape
    assert np.all(gtv_imgs.shape == target_shape), gtv_imgs.shape
    return t1_imgs, gtv_imgs


def remove_unannotated_sections(X: tf.Tensor, y: tf.Tensor, n_min: int = 10) -> tuple:
    """
    Remove z-sections without a min amount of cancer.
    X: n_sections, x, y, 1
    """
    X = X.numpy()
    y = y.numpy()
    assert X.ndim == 4, f"{X.ndim}"  # should be z, x, y, 1. If it's 5, there might be an extra batch dim.
    assert X.shape[-1] == 1, f"{X.shape}"  # last dim should be channels, of which there should be only 1
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
    new_shape = X.shape
    return X, y, new_shape




