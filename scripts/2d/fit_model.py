import datetime
import os
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.two_dims import get_simple_cnn
from cancer_ml.models.loss import DiceBCELoss

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/flattened")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/")
TB_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/tb_runs/")
FILTER_SIZES = [16, 32]
N_EPOCHS = 50
BATCH_SIZE = 64

# load data
dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    dsets[name] = ds


def change_dtype(some_X, some_y) -> tuple:
    some_y = tf.cast(some_y, tf.float32)
    return some_X, some_y


train_ds = dsets["train"].map(change_dtype).batch(BATCH_SIZE)
val_ds = dsets["val"].map(change_dtype).batch(BATCH_SIZE)

#
X, y = next(iter(dsets["train"].take(1)))
input_shape = X.shape
print(f"{X.shape=}")

# make model
model = get_simple_cnn(
    input_shape,
    filter_sizes=FILTER_SIZES,
)

# compile
optimizer = keras.optimizers.Adam()
loss_fn = DiceBCELoss()
metrics = [keras.metrics.BinaryIoU(), keras.losses.Dice(), keras.losses.BinaryCrossentropy()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)


# fit
model_id = f"flattened"
model_dir = OUTPUT / "2d" / model_id
os.makedirs(model_dir, exist_ok=True)
model_file = model_dir / "cnn.weights.h5"
csv_file = model_dir / "log.csv"
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tb_folder = TB_FOLDER / "2d" / date_str
callbacks = [
    keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
    keras.callbacks.CSVLogger(csv_file),
    keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    keras.callbacks.EarlyStopping(patience=10),
]
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=N_EPOCHS,
    callbacks=callbacks,
)

