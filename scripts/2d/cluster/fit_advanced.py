"""
Fit model on HPC.
"""
import datetime
import os

import keras
import tensorflow as tf

from cancer_ml.models.two_dims.custom import get_advanced_cnn
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import assert_gpu_available, get_args_dirs
from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.params import get_data_params

# paths
FILTER_SIZES = [32, 64]
DROPOUT_RATE = 0.1
ADD_SKIP_CONNECTIONS = True
BATCH_SIZE = 64
N_EPOCHS = 50


def main() -> None:
    # make sure we are using GPU
    assert_gpu_available()

    # arguments
    data_dir, output_dir, tb_dir = get_args_dirs()

    # load data
    print("---Load---")
    def change_dtype(t1_imgs, gtv_imgs) -> tuple:
        """We want X as float16 (to save memory) and y as float32 (to have good loss calculations)"""
        t1_imgs = tf.cast(t1_imgs, tf.float16)
        gtv_imgs = tf.cast(gtv_imgs, tf.float32)
        return t1_imgs, gtv_imgs

    dsets = {}
    for name in ["train", "val", "test"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        ds = ds.map(change_dtype).batch(BATCH_SIZE)
        dsets[name] = ds

    X, y = next(iter(dsets["train"].take(1)))
    hparams = get_data_params(X)
    input_shape = X.shape[1:]
    print(f"{X.shape=}")

    # get model
    model = get_advanced_cnn(
        input_shape=input_shape,
        filter_sizes=FILTER_SIZES,
        dropout_rate=DROPOUT_RATE,
        add_skips=ADD_SKIP_CONNECTIONS,
    )

    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    # prep output
    print("---Prepping output---")
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / date_str
    os.makedirs(model_dir, exist_ok=True)

    # set up callbacks
    tb_folder = tb_dir / "2d" / date_str
    callbacks = [
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
        keras.callbacks.EarlyStopping(patience=10),
    ]

    # save hyperparameters
    hparams["model_type"] = "advanced"
    hparams["filter_sizes"] = FILTER_SIZES
    hparams["dropout_rate"] = DROPOUT_RATE
    hparams["n_epochs"] = N_EPOCHS
    hparams["batch_size"] = BATCH_SIZE
    hparams["add_skips"] = ADD_SKIP_CONNECTIONS

    fit_and_evaluate(
        model=model,
        dsets=dsets,
        model_dir=model_dir,
        callbacks=callbacks,
        hparams=hparams,
        verbose=2,
    )

if __name__ == "__main__":
    main()