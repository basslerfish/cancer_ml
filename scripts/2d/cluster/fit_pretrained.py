"""
Let's fit a pretrained segmentation model to our data.
Now on the cluster.
"""
import os
import datetime

import keras
import numpy as np
import tensorflow as tf

from cancer_ml.models.two_dims.pretrained import get_pretrained_deeplab, dl_unfreeze_aspp_decoder
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import get_args_dirs
from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.params import get_data_params

# params
N_EPOCHS = 100
BATCH_SIZE = 128


def main() -> None:
    # get paths
    data_dir, output_dir, tb_dir = get_args_dirs()

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

        # change dtype
        some_y = tf.cast(some_y, tf.float32)
        return some_X, some_y


    dsets = {}
    for name in ["train", "val", "test"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        ds = ds.map(preprocess).batch(BATCH_SIZE)
        dsets[name] = ds

    X, y = next(iter(dsets["train"].take(1)))
    hparams = get_data_params(X)
    assert X.ndim == 4  # n_batch, x, y, n_channels
    assert X.shape[-1] == 3  # last dim should be channels
    image_size = (X.shape[1], X.shape[2])

    # load model
    print("---Load model---")
    model = get_pretrained_deeplab()
    model.preprocessor.image_converter.image_size = image_size
    model = dl_unfreeze_aspp_decoder(model, also_batch_norm=True)

    # build model
    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice()]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    # prep output
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / date_str
    os.makedirs(model_dir, exist_ok=True)
    tb_folder = tb_dir / "2d" / date_str
    callbacks = [
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    ]
    hparams["model_type"] = "pretrained_deeplab"
    hparams["batch_size"] = BATCH_SIZE
    hparams["n_epochs"] = N_EPOCHS

    # go
    fit_and_evaluate(
        model=model,
        dsets=dsets,
        model_dir=model_dir,
        callbacks=callbacks,
        verbose=2,
        hparams=hparams
    )


if __name__ == "__main__":
    main()