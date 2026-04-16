"""
Fit a custom CNN to predict segmentation mask on 2D images.
"""
import datetime
import os

import keras
import tensorflow as tf
import yaml

from cancer_ml.models.two_dims.cnn.custom import get_advanced_cnn
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.utils import get_data_info, get_param_count
from cancer_ml.models.training import fit_and_evaluate
from cancer_ml.paths import get_arg_paths

# set paths & load config
paths = get_arg_paths()
assert paths["data"].is_dir()
assert paths["config"].is_file()
with open(paths["config"], "r") as file:
    config = yaml.safe_load(file)

# load data
def change_dtype(some_X, some_y) -> tuple:
    """Change y dtype to float32"""
    some_y = tf.cast(some_y, tf.float32)
    return some_X, some_y

dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(paths["data"] / name))
    ds = ds.map(change_dtype).batch(config["training"]["batch_size"])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    dsets[name] = ds

# get basic info
data_info = get_data_info(dsets)
config["data"] = data_info
print(f"Batch shape: {config['data']['batch_shape']}")

# make model
model = get_advanced_cnn(
    data_info["batch_shape"],
    filter_sizes=config["model"]["filter_sizes"],
    dropout_rate=config["model"]["dropout"],
    add_skips=config["model"]["add_skips"],
)

# compile
optimizer = keras.optimizers.Adam(
    learning_rate=config["training"]["learning_rate"],
)
loss_fn = DiceBCELoss()
metrics = [keras.losses.Dice()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)
weight_counts = get_param_count(model)
print(f"Trainable weights: {weight_counts['trainable_weights']:,}")
print(f"Non-trainable weights: {weight_counts['non_trainable_weights']:,}")

# prepare output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = f"cnn_{date_str}"
config["meta"] = {"model_id": model_id, **weight_counts}
print(f"Model ID: {model_id}")

model_dir = paths["output"] / "2d" / model_id
os.makedirs(model_dir, exist_ok=True)
paths["model"] = model_dir
callbacks = [
    keras.callbacks.EarlyStopping(patience=config["training"]["patience"]),
]

fit_and_evaluate(
    model=model,
    config=config,
    paths=paths,
    callbacks=callbacks,
    dsets=dsets,
)
