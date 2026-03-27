from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.three_dims import get_simple_cnn

# paths
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/samples500_val15_test15_128-128-64")
FILTER_SIZES = [16, 32]
BATCH_SIZE = 4

# get shape
dset_name = DSET_FOLDER.name
dset_shape = dset_name.split("_")[-1]
dset_shape = dset_shape.split("-")
dset_shape = [int(x) for x in dset_shape]
print(dset_shape)

# load data
train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
val_ds = tf.data.Dataset.load(str(DSET_FOLDER / "val"))

# change dtype
def change_dtype(t1_imgs, gtv_imgs) -> tuple:
    t1_imgs = tf.cast(t1_imgs, tf.float16)
    gtv_imgs = tf.cast(gtv_imgs, tf.float32)
    return t1_imgs, gtv_imgs


train_ds = train_ds.map(change_dtype)
val_ds = val_ds.map(change_dtype)
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# get model
model = get_simple_cnn(
    input_shape=dset_shape,
    filter_sizes=FILTER_SIZES,
)

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.Dice()
metrics = [keras.metrics.BinaryIoU()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics,
)

callbacks = []
model.fit(
    train_ds,
    validation_data=val_ds,
)