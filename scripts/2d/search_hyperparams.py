"""
Let's search for best model params.
"""
from pathlib import Path

import keras
import keras_tuner as kt
import tensorflow as tf

from cancer_ml.models.two_dims import get_flexible_model
from cancer_ml.models.loss import DiceBCELoss

# params
MAX_TRIALS = 50
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/params_search")
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/flattened")
BATCH_SIZE = 64
N_EPOCHS = 50

# load data
print("---Load data---")
dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
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
    model = get_flexible_model(
        input_shape=input_shape,
        model_type=hp.Choice("model_type", ["simple", "advanced"]),
        filter_sizes=hp.Choice("model_type", possible_fs),
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
    objective="val_loss",
    max_trials=MAX_TRIALS,
    directory=OUTPUT,
    project_name="optimize_2d",
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=N_EPOCHS,
)