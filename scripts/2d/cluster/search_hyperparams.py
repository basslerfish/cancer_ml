"""
Let's search for best model params.
"""
import argparse
import os
from pathlib import Path

import keras
import keras_tuner as kt
import tensorflow as tf

from cancer_ml.models.two_dims.custom import get_flexible_model
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import assert_gpu_available

# params
MAX_TRIALS = 50
BATCH_SIZE = 64
N_EPOCHS = 50
FILTER_SIZES = {
    "16-32": [16, 32],
    "16-32-64": [16, 32, 64],
    "32-64": [32, 64],
    "32-64-128": [32, 64, 128]
}


def main() -> None:
    # make sure we are using GPU
    assert_gpu_available()

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    # paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    assert data_dir.is_dir(), f"{data_dir} does not exist"
    assert output_dir.is_dir(), f"{output_dir} does not exist"
    output_dir = output_dir / "params_search"
    os.makedirs(output_dir, exist_ok=True)

    # load data
    print("---Load data---")
    dsets = {}
    for name in ["train", "val", "test"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        dsets[name] = ds


    def change_dtype(some_X, some_y) -> tuple:
        some_y = tf.cast(some_y, tf.float32)
        return some_X, some_y


    train_ds = dsets["train"].map(change_dtype).batch(BATCH_SIZE)
    val_ds = dsets["val"].map(change_dtype).batch(BATCH_SIZE)
    X, y = next(iter(dsets["train"].take(1)))
    input_shape = X.shape


    print("---Run optimization---")
    # define possible filter sizes
    possible_fs = []
    possible_fs.append([16, 32])
    possible_fs.append([32, 64])
    possible_fs.append([32, 64, 128])

    # func
    def build_model(hp: kt.HyperParameters) -> keras.Model:
        fs_name = hp.Choice("filter_sizes", list(FILTER_SIZES.keys()))
        filter_sizes = FILTER_SIZES[fs_name]
        model = get_flexible_model(
            input_shape=input_shape,
            model_type=hp.Choice("model_type", ["simple", "advanced"]),
            filter_sizes=filter_sizes,
            add_skips=hp.Choice("add_skips", [False, True])
        )
        optimizer = keras.optimizers.Adam()
        loss_fn = DiceBCELoss()
        metrics = [keras.losses.Dice(), keras.losses.BinaryCrossentropy()]
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )
        return model

    # go!
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective="val_dice",
        max_trials=MAX_TRIALS,
        directory=output_dir,
        project_name="optimize_2d",
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=N_EPOCHS,
        verbose=2,
    )

if __name__ == "__main__":
    main()