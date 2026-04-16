"""
Fit a very simple CNN to predict segmentation mask on 2D images.
"""
import datetime
import os
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.params import get_data_params, write_hparams
from cancer_ml.models.two_dims.custom import get_simple_cnn

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_zscore_val15_test15_128-128")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/")
TB_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/tb_runs/")
FILTER_SIZES = [16, 32]
N_EPOCHS = 50
BATCH_SIZE = 64


# load data
def change_dtype(some_X, some_y) -> tuple:
    """Change y dtype to float32"""
    some_y = tf.cast(some_y, tf.float32)
    return some_X, some_y

dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    ds = ds.map(change_dtype).batch(BATCH_SIZE)
    dsets[name] = ds


# get basic info
X, y = next(iter(dsets["train"].take(1)))
hparams = get_data_params(X)
input_shape = X.shape[1:]
print(f"{X.shape=}")

# make model
model = get_simple_cnn(
    input_shape,
    filter_sizes=FILTER_SIZES,
)


# compile
optimizer = keras.optimizers.Adam()
loss_fn = DiceBCELoss()
metrics = [keras.losses.Dice()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)

# prepare output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = OUTPUT / "2d" / date_str
os.makedirs(model_dir, exist_ok=True)
csv_file = model_dir / "log.csv"
tb_folder = TB_FOLDER / "2d" / date_str
callbacks = [
    keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    keras.callbacks.EarlyStopping(patience=10),
]

# save hyperparameters
hparams["model_type"] = "simple"
hparams["filter_sizes"] = FILTER_SIZES
hparams["n_epochs"] = N_EPOCHS
hparams["batch_size"] = BATCH_SIZE
write_hparams(hparams, model_dir / "hparams.json")

# fit
fit_and_evaluate(
    model=model,
    model_dir=model_dir,
    dsets=dsets,
    callbacks=callbacks,
    hparams=hparams
)