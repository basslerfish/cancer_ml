"""
Fit a pretrained vision transformer on the cancer segmentation task.

TODO: consider learning rate scheduler
"""
import datetime
import os

import keras.losses
import tensorflow as tf
import yaml

from cancer_ml.models.two_dims.transformer.pretrained import get_pretrained_model, unfreeze_final, unfreeze_post_encoder
from cancer_ml.paths import get_arg_paths
from cancer_ml.models.utils import get_data_info, get_param_count, get_recursive_description
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.training import fit_and_evaluate, unfreeze_all

# set paths
paths = get_arg_paths()
print(paths)
with open(paths["config"], "r") as file:
    config = yaml.safe_load(file)

# load data
def change_dtype(some_X, some_y) -> tuple:
    """Change y dtype to float32"""
    some_X = tf.image.grayscale_to_rgb(some_X)
    some_X = tf.cast(some_X, tf.float32)
    some_X = some_X / 255.0
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
print(f"Batch shape: {data_info['batch_shape']}")

# go!
model = get_pretrained_model()
unfreeze_mode = config["training"]["unfreeze"]
print(f"Unfreeze mode: {unfreeze_mode}")
if unfreeze_mode == "all":
    model = unfreeze_all(model)
elif unfreeze_mode == "decoder":
    model = unfreeze_post_encoder(model)
elif unfreeze_mode == "final":
    model = unfreeze_final(model)
else:
    raise ValueError(f"{unfreeze_mode=} unknown.")

print("---Model layers---")
get_recursive_description(model)
optimizer = keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
model.compile(
    optimizer=optimizer,
    loss=DiceBCELoss(),
    metrics=[keras.losses.Dice()],
)
param_counts = get_param_count(model)
print(f"Trainable weights: {param_counts['trainable_weights']:,}")
print(f"Non-trainable weights: {param_counts['non_trainable_weights']:,}")

# prep output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = f"pretrained_vit_{date_str}"
config["meta"] = {"model_id": model_id, **param_counts}
print(f"Model ID: {model_id}")
model_dir = paths["output"] / "2d" / model_id
paths["model"] = model_dir
os.makedirs(model_dir)

print("---Fit---")
fit_and_evaluate(
    model=model,
    dsets=dsets,
    config=config,
    paths=paths,
    verbose=2,
)