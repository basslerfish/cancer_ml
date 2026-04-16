"""
Let's fit a pretrained segmentation CNN model to our data.
We again use the Deeplabv3+.

We gradually unfreeze more parts of the model.
"""
import os
import datetime

import keras
import tensorflow as tf
import wandb
import yaml
from wandb.integration.keras import WandbMetricsLogger

from cancer_ml.models.utils import get_param_count, get_data_info
from cancer_ml.models.two_dims.cnn.pretrained import get_pretrained_deeplab, unfreeze_aspp_decoder, unfreeze_last
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.paths import get_arg_paths
from cancer_ml.models.training import unfreeze_all

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
model.preprocessor.image_converter.image_size = (batch_shape[1], batch_shape[2])
loss_fn = DiceBCELoss()
metrics = [keras.losses.Dice()]

# prep output
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_id = f"deeplab_mp_{date_str}"
config["meta"] = {"model_id": model_id}
print(f"Model ID: {model_id}")
model_dir = paths["output"] / "2d" / model_id
paths["model"] = model_dir
os.makedirs(model_dir, exist_ok=True)
csv_file = model_dir / "log.csv"
best_weights_file = model_dir / "best.weights.h5"
final_weights_file = model_dir / "final.weights.h5"

# init wandb
wandb.init(
    project="cancer_ml",
    config=config,
    name=config["meta"]["model_id"],
    dir=paths["wandb"].parent,
)

# callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        best_weights_file,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_dice"),
    keras.callbacks.CSVLogger(csv_file),
    WandbMetricsLogger(),
]

phases_unfreeze = config["training"]["phases_unfreeze"]
phase_lrs = config["training"]["phases_lr"]
phase_epochs = config["training"]["phases_epochs"]
current_epoch = 0
for i_phase in range(len(phases_unfreeze)):
    this_unfreeze = phases_unfreeze[i_phase]
    this_lr = phase_lrs[i_phase]
    this_epochs = phase_epochs[i_phase]

    print(f"---Fit: Phase {i_phase} {this_unfreeze}---")
    if this_unfreeze == "final":  # only final layer
        model = unfreeze_last(model)
    elif this_unfreeze == "aspp":  # aspp and decoder
        model = unfreeze_aspp_decoder(model, also_batch_norm=True)
    elif this_unfreeze == "all":
        model = unfreeze_all(model)
    else:
        raise ValueError(f"{this_unfreeze=} unknown")

    optimizer = keras.optimizers.Adam(
        learning_rate=this_lr)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    weight_counts = get_param_count(model)
    print(f"Trainable weights: {weight_counts['trainable_weights']:,}")
    print(f"Non-trainable weights: {weight_counts['non_trainable_weights']:,}")

     # go!
    config["training"]["learning_rate"] = this_lr
    config["training"]["epochs"] = this_epochs + current_epoch

    model.fit(
        dsets["train"],
        validation_data=dsets["val"],
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        initial_epoch=current_epoch,
        verbose=2,
    )
    model.save_weights(final_weights_file)
    current_epoch += phase_epochs[i_phase]


model.load_weights(best_weights_file)
scores = model.evaluate(
    dsets["test"],
    verbose=2,
)
scores = {
    "test_dice": scores[0],
}
wandb.log(scores)
wandb.finish()
