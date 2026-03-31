"""
Let's fit a pretrained segmentation model to our data.
Now on the cluster.
"""
import os
import datetime

import keras
import tensorflow as tf

from cancer_ml.models.two_dims.pretrained import get_pretrained_deeplab, dl_unfreeze_aspp_decoder
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.utils import get_args_dirs

# params
N_EPOCHS = 100
BATCH_SIZE = 64


def main() -> None:
    # get paths
    data_dir, output_dir, tb_dir = get_args_dirs()

    # load data
    print("---Load data---")
    dsets = {}
    for name in ["train", "val"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        dsets[name] = ds


    def preprocess(some_X: tf.Tensor, some_y: tf.Tensor) -> tuple:
        """
        Change X to 3 channels (required for Resnet, which is the backbone here)
        Change y to float32 (required for loss calc)
        """
        # add channels to X
        some_X = tf.image.grayscale_to_rgb(some_X)

        # change dtype
        some_y = tf.cast(some_y, tf.float32)
        return some_X, some_y


    train_ds = dsets["train"].map(preprocess).batch(BATCH_SIZE)
    val_ds = dsets["val"].map(preprocess).batch(BATCH_SIZE)

    X, y = next(iter(train_ds.take(1)))
    print(f"{X.shape=}")
    print(f"{y.shape=}")
    assert X.shape[-1] == 3

    # load model
    print("---Load model---")
    model = get_pretrained_deeplab()
    model.preprocessor.image_converter.image_size = (128, 128)
    model = dl_unfreeze_aspp_decoder(model)

    print("---Fitting model---")
    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice()]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    model_dir = output_dir / "2d" / "pretrained_deeplabv3+"
    os.makedirs(model_dir, exist_ok=True)
    model_file = model_dir / "cnn.weights.h5"
    csv_file = model_dir / "log.csv"
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_folder = tb_dir / "2d_pretrained" / date_str
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
        keras.callbacks.CSVLogger(csv_file),
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=N_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

if __name__ == "__main__":
    main()