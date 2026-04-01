"""
Let's fit a pretrained segmentation model to our data.

We do a two phase training - in the first phase, we leave the encoder backbone untouched.
"""
import os
import datetime

import keras
import tensorflow as tf

from cancer_ml.models.two_dims.pretrained import get_pretrained_deeplab, dl_unfreeze_aspp_decoder
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import get_args_dirs
from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.params import get_data_params

# params
N_EPOCHS = 100
BATCH_SIZE = 64
EPOCHS_TO_UNFREEZE = 20
FIRST_LR = 10 ** -3
SECOND_LR = 10 ** -4


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

    # load model
    print("---Load model---")
    model = get_pretrained_deeplab()
    model.preprocessor.image_converter.image_size = (X.shape[1], X.shape[2])
    print("Unfreezing weights except for encoder ResNet.")
    model = dl_unfreeze_aspp_decoder(model, also_batch_norm=True)

    print("---First fit---")
    optimizer = keras.optimizers.Adam(FIRST_LR)
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice()]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    model.fit(
        dsets["train"],
        validation_data=dsets["val"],
        epochs=EPOCHS_TO_UNFREEZE,
        verbose=2,
    )

    print("---Second fit---")
    model.trainable = True
    print("All weights are trainable now.")

    # define some callbacks
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / date_str
    os.makedirs(model_dir, exist_ok=True)
    tb_folder = tb_dir / "2d" / date_str
    callbacks = [
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
        keras.callbacks.EarlyStopping(patience=33),
    ]

    # get a new optimizer with a different lr
    hparams["model_type"] = "pretrained_deeplab"
    hparams["comment"] = "two_phase_training"
    hparams["phase1_lr"] = FIRST_LR
    hparams["phase2_lr"] = SECOND_LR

    optimizer = keras.optimizers.Adam(learning_rate=SECOND_LR)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )

    fit_and_evaluate(
        model=model,
        dsets=dsets,
        hparams=hparams,
        model_dir=model_dir,
        verbose=2,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()