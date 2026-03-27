"""
Fit model.
"""
import argparse
import datetime
import os
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.three_dims import get_simple_cnn, DiceBCELoss
from cancer_ml.utils import assert_gpu_available

# paths
FILTER_SIZES = [32, 64]
BATCH_SIZE = 4
N_EPOCHS = 50

def main() -> None:
    # make sure we are using GPU
    assert_gpu_available()

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--tb_dir")
    args = parser.parse_args()

    # check paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    tb_dir = Path(args.tb_dir)
    assert data_dir.is_dir(), f"{data_dir} does not exist"
    assert output_dir.is_dir(), f"{output_dir} does not exist"
    assert tb_dir.is_dir(), f"{tb_dir} does not exist"

    # get shape
    dset_name = data_dir.name
    dset_shape = dset_name.split("_")[-1]
    dset_shape = dset_shape.split("-")
    dset_shape = [int(x) for x in dset_shape]
    dset_shape = [dset_shape[2], dset_shape[0], dset_shape[1], 1]
    print(dset_shape)

    # load data
    print("---Load---")
    train_ds = tf.data.Dataset.load(str(data_dir / "train"))
    val_ds = tf.data.Dataset.load(str(data_dir / "val"))

    # change dtype
    def change_dtype(t1_imgs, gtv_imgs) -> tuple:
        """We want X as float16 (to save memory) and y as float32 (to have good loss calculations)"""
        t1_imgs = tf.cast(t1_imgs, tf.float16)
        gtv_imgs = tf.cast(gtv_imgs, tf.float32)
        return t1_imgs, gtv_imgs


    train_ds = train_ds.map(change_dtype)
    val_ds = val_ds.map(change_dtype)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    for X, y in train_ds.take(1):
        print(f"Batch shape: {X.shape}")

    # get model
    model = get_simple_cnn(
        input_shape=dset_shape,
        filter_sizes=FILTER_SIZES,
    )

    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.metrics.BinaryIoU(), keras.losses.Dice]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    # prep output
    print("---Prepping output---")
    filters_str = f"-".join([str(x) for x in FILTER_SIZES])
    dimensions_str = f"-".join([str(x) for x in dset_shape])
    model_id = f"{filters_str}_{dimensions_str}"
    print(f"Model ID: {model_id}")
    os.makedirs(output_dir / model_id, exist_ok=True)
    model_file = output_dir / model_id / f"cnn.weights.h5"
    csv_file = output_dir / model_id / "log.csv"
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_folder = tb_dir / "3d" / date_str
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
        keras.callbacks.CSVLogger(csv_file),
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
        keras.callbacks.EarlyStopping(patience=10),
    ]

    print("---Fitting model---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=N_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

if __name__ == "__main__":
    main()