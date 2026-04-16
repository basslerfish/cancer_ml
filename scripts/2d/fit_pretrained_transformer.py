"""
Fit a pretrained vision transformer on the cancer segmentation task.

TODO: unfreeze some layers, then train!
"""
import tensorflow as tf
import yaml

from cancer_ml.models.two_dims.transformer.pretrained import get_pretrained_model
from cancer_ml.paths import get_arg_paths
from cancer_ml.models.params import get_data_params

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
    dsets[name] = ds

# get basic info
X, y = next(iter(dsets["train"].take(1)))
input_shape = X.shape[1:]
print(get_data_params(X))
print(f"{X.shape=}")

# go!
model = get_pretrained_model()
model(X)

