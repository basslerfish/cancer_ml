"""
Fit a custom CNN to predict segmentation mask on 2D images.
"""
import datetime
import os

import keras
import tensorflow as tf

from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.training import fit_and_evaluate
from cancer_ml.models.two_dims.cnn.basic import get_simple_cnn
from cancer_ml.models.utils import get_data_info, get_param_count, load_config_yaml
from cancer_ml.paths import get_arg_paths
from cancer_ml.models.callbacks import PredictionPlotCallback

# set paths & load config
paths = get_arg_paths()
assert paths["data"].is_dir()
assert paths["config"].is_file()
config = load_config_yaml(paths["config"])

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
data_info = get_data_info(dsets)
config["data"] = data_info

# make model
model = get_simple_cnn(
    data_info["batch_shape"],
    filter_sizes=config["model"]["filter_sizes"],
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

# prepare output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = f"basic_cnn_{date_str}"
if "meta" not in config.keys():
    config["meta"] = {}
config["meta"].update({"model_id": model_id, **weight_counts})
print(f"Model ID: {model_id}")

model_dir = paths["output"] / "2d" / model_id
os.makedirs(model_dir, exist_ok=True)
paths["model"] = model_dir
callbacks = [
    PredictionPlotCallback(
        model=model,
        output_folder=model_dir,
        dset=dsets["val"],
        every_n_epoch=1,
        use_wandb=config["meta"]["use_wandb"],
    )
]

# go!
fit_and_evaluate(
    model=model,
    config=config,
    paths=paths,
    callbacks=callbacks,
    dsets=dsets,
)
