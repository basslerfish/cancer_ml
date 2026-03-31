"""
Fit model.
"""
import datetime
import os
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.three_dims.custom import get_simple_cnn
from cancer_ml.models.loss import DiceBCELoss

# paths
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/samples500_val15_test15_128-128-64")
FILTER_SIZES = [16, 32]
BATCH_SIZE = 4
N_EPOCHS = 50
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/3d")
TB_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/tb_runs")


# get shape
dset_name = DSET_FOLDER.name
dset_shape = dset_name.split("_")[-1]
dset_shape = dset_shape.split("-")
dset_shape = [int(x) for x in dset_shape]
dset_shape = [dset_shape[2], dset_shape[0], dset_shape[1], 1]
print(dset_shape)

# load data
print("---Load---")
train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
val_ds = tf.data.Dataset.load(str(DSET_FOLDER / "val"))

# change dtype
def change_dtype(t1_imgs, gtv_imgs) -> tuple:
    """We want X as float16 (to save memory) and y as float32 (to have good loss calculations)"""
    t1_imgs = tf.cast(t1_imgs, tf.float16)
    gtv_imgs = tf.cast(gtv_imgs, tf.float32)
    return t1_imgs, gtv_imgs


train_ds = train_ds.map(change_dtype)
val_ds = val_ds.map(change_dtype)
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)
for X, y in train_ds.take(1):
    print(f"Batch shape: {X.shape}")

# get model
model = get_simple_cnn(
    input_shape=dset_shape,
    filter_sizes=FILTER_SIZES,
)

optimizer = keras.optimizers.Adam()
loss_fn = DiceBCELoss()
metrics = [keras.metrics.BinaryIoU(), keras.losses.Dice]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics,
)
# prep output
print("---Prepping output---")
filters_str = f"-".join([str(x) for x in FILTER_SIZES])
dimensions_str = f"-".join([str(x) for x in dset_shape])
model_id = f"{filters_str}_{dimensions_str}"
print(f"Model ID: {model_id}")
os.makedirs(OUTPUT / model_id, exist_ok=True)
model_file = OUTPUT / model_id / f"cnn.weights.h5"
csv_file = OUTPUT / model_id / "log.csv"
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tb_folder = TB_DIR / "3d" / date_str
callbacks = [
    keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
    keras.callbacks.CSVLogger(csv_file),
    keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    keras.callbacks.EarlyStopping(patience=10),
]

print("---Fitting model---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=N_EPOCHS,
    callbacks=callbacks,
)