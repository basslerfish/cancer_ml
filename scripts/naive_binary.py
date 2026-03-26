"""
Train a very simple CNN to predict whether a T1 slice contains cancer or not.
This is not very useful, but computationally less expensive than segmentation.
"""
import datetime
from pathlib import Path

import keras
import tensorflow as tf

from src.model import get_binary_cnn

# paths
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/train_100")
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/fits")
TB_OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/tb_runs")
BATCH_SIZE = 16
FILTER_SIZES = [32, 64, 128]
EPOCHS = 50
VAL_FRAC = 0.1
N_SAMPLES = 100

# load data and sample info
print("---Load---")
ds = tf.data.Dataset.load(str(DSET_FOLDER))
for X, y in ds.take(1):
    batch_shape = X.shape
print(f"Image shape: {batch_shape}")
if len(batch_shape) == 4:
    input_shape = batch_shape[1:]
else:
    input_shape = batch_shape

# convert y so that for each T1 image in a stack, we simply check whether there is cancer or not
print("---Convert---")
def convert_y(some_X: tf.Tensor, some_y: tf.Tensor) -> tuple:
    """
    Convert y to binary for each image in T1 frame.
    Binary as in: cancer annotated or not
    """
    some_y = tf.cast(some_y, tf.bool)
    if some_y.ndim == 4:
        some_y = tf.reduce_any(some_y, axis=(1, 2))
    elif some_y.ndim == 3:
        some_y = tf.reduce_any(some_y, axis=(0, 1))
    else:
        raise ValueError(f"{some_y.ndim=}")
    some_y = tf.cast(some_y, tf.float32)
    return some_X, some_y


ds = ds.map(convert_y)

print("---Split---")
ds = ds.shuffle(ds.cardinality())
train_samples = int(N_SAMPLES * (1 - VAL_FRAC))
val_samples = N_SAMPLES - train_samples
print(f"Train samples: {train_samples}")
print(f"Val samples: {val_samples}")
train_ds = ds.take(train_samples)
val_ds = ds.skip(train_samples)
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

for X, y in train_ds.take(1):
    print("train", X.shape, y.shape)
for X, y in val_ds.take(1):
    print("val", X.shape, y.shape)
    print(f"val")

print("---Compile---")
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy()
metrics = [keras.metrics.BinaryAccuracy()]
model = get_binary_cnn(input_shape, FILTER_SIZES)
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics,
)
print(model.summary(80))


model_file = OUTPUT / "model.weights.h5"
csv_file = OUTPUT / "model.csv"
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tb_folder = TB_OUTPUT / "naive_binary" / date_str
callbacks = [
    keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
    keras.callbacks.CSVLogger(csv_file),
    keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
#   keras.callbacks.EarlyStopping(patience=10),
]

print("---Fit---")
model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
)
