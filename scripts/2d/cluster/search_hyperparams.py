"""
Let's search for best model params for an advanced CNN.
We can modify
- depth / filter_sizes
- add_skips
- dropout_rate
"""
import os

import keras
import keras_tuner as kt
import tensorflow as tf

from cancer_ml.models.two_dims.custom import get_advanced_cnn
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import assert_gpu_available, get_args_dirs

# params
MAX_TRIALS = 50
BATCH_SIZE = 64
N_EPOCHS = 50
FILTER_SIZES = {
    "16-32-64": [16, 32, 64],
    "32-64-128": [32, 64, 128],
    "32-64-128-256": [32, 64, 128, 256],
    "64-128-256-512": [64, 128, 256, 512]
}
ADD_SKIPS = [False, True]
DROPOUT_RATES = [0.1, 0.3, 0.5]

def main() -> None:
    # make sure we are using GPU
    assert_gpu_available()

    # arguments
    data_dir, output_dir = get_args_dirs(also_tb=False)
    os.makedirs(output_dir, exist_ok=True)

    # load data
    print("---Load data---")
    def change_dtype(some_X, some_y) -> tuple:
        """Change dtype of y to float32 (required for loss calc)"""
        some_y = tf.cast(some_y, tf.float32)
        return some_X, some_y

    dsets = {}
    for name in ["train", "val"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        ds = ds.map(change_dtype).batch(BATCH_SIZE)
        dsets[name] = ds

    X, y = next(iter(dsets["train"].take(1)))
    input_shape = X.shape[1:]

    print("---Run optimization---")
    # func
    def build_model(hp: kt.HyperParameters) -> keras.Model:
        fs_name = hp.Choice("filter_sizes", list(FILTER_SIZES.keys()))
        filter_sizes = FILTER_SIZES[fs_name]
        model = get_advanced_cnn(
            input_shape=input_shape,
            filter_sizes=filter_sizes,
            add_skips=hp.Choice("add_skips", ADD_SKIPS),
            dropout_rate=hp.Choice("dropout_rate", DROPOUT_RATES),
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
        project_name="optimize_advanced",
    )

    tuner.search(
        dsets["train"],
        validation_data=dsets["val"],
        epochs=N_EPOCHS,
        verbose=2,
    )

if __name__ == "__main__":
    main()