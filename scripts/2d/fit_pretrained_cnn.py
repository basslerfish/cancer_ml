"""
Let's fit a pretrained segmentation model to our data.
We use DeepLabv3+ here. This is an encoder-decoder CNN with a special atrous spatial pyramid pooling (ASPP) head.
We only unfreeze the ASPP head and the decoder, leaving the encoder unaffected.
"""
import os
import datetime

import keras
import tensorflow as tf
import yaml

from cancer_ml.models.utils import get_param_count, get_data_info
from cancer_ml.models.two_dims.cnn.pretrained import get_pretrained_deeplab, unfreeze_last, unfreeze_aspp_decoder
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.training import fit_and_evaluate, unfreeze_all
from cancer_ml.paths import get_arg_paths

# get paths & config
paths = get_arg_paths()
assert paths["data"].is_dir()
assert paths["config"].is_file()
with open(paths["config"], "r") as file:
    config = yaml.safe_load(file)

# load data
print("---Load data---")
def preprocess(some_X: tf.Tensor, some_y: tf.Tensor) -> tuple:
    """
    Change X to 3 channels (required for Resnet, which is the backbone here)
    Change y to float32 (required for loss calc)
    """
    # add channels to X
    some_X = tf.image.grayscale_to_rgb(some_X)
    some_X = tf.cast(some_X, tf.float32)
    some_X = some_X / 255.0

    # change dtype
    some_y = tf.cast(some_y, tf.float32)
    return some_X, some_y


dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(paths["data"] / name))
    ds = ds.map(preprocess).batch(config["training"]["batch_size"])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    dsets[name] = ds


# get basic info
data_info = get_data_info(dsets)
config["data"] = data_info
batch_shape = data_info["batch_shape"]
assert len(batch_shape) == 4
print(f"Batch shape: {batch_shape}")

# load model
print("---Load model---")
model = get_pretrained_deeplab()
if config["unfreeze"] == "all":
    model = unfreeze_all(model)
elif config["unfreeze"] == "aspp":
    model = unfreeze_aspp_decoder(model)
elif config["unfreeze"] == "final":
    model = unfreeze_last(model)
else:
    raise ValueError(f"{config['unfreeze']}")

model.preprocessor.image_converter.image_size = (batch_shape[1], batch_shape[2])
optimizer = keras.optimizers.Adam()
loss_fn = DiceBCELoss()
metrics = [keras.losses.Dice()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics,
)
weight_counts = get_param_count(model)
print(f"Trainable weights: {weight_counts['trainable_weights']:,}")
print(f"Non-trainable weights: {weight_counts['non_trainable_weights']:,}")

# prep output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = f"deeplab_{date_str}"
config["meta"] = {"model_id": model_id, **weight_counts}
print(f"Model ID: {model_id}")
model_dir = paths["output"] / "2d" / model_id
paths["model"] = model_dir
os.makedirs(model_dir, exist_ok=True)
callbacks = []
 # go!
fit_and_evaluate(
    model=model,
    dsets=dsets,
    config=config,
    paths=paths,
    callbacks=callbacks,
)
