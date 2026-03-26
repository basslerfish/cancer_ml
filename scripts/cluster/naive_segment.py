"""
Train very simple CNN on HPC.
"""
import argparse
import datetime
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.model import get_simple_cnn
from cancer_ml.utils import assert_gpu_available

# paths
BATCH_SIZE = 4
FILTER_SIZES = [32, 64, 128]
EPOCHS = 50
VAL_FRAC = 0.1
N_SAMPLES = 100

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
    print("---Check paths---")
    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"{data_dir} does not exist."
    output_dir = Path(args.output_dir)
    assert output_dir.is_dir(), f"{output_dir} does not exist."
    tb_dir = Path(args.tb_dir)

    # load data and sample info
    print("---Load---")
    ds = tf.data.Dataset.load(str(data_dir))
    for X, y in ds.take(1):
        data_shape = X.shape
    print(f"Image shape: {data_shape}")
    assert len(data_shape) == 4  # n_img, x, y, channels
    assert data_shape[3] == 1  #  we want a single channel

    print("---Convert---")
    def convert_y(some_X: tf.Tensor, some_y: tf.Tensor) -> tuple:
        """
        We want y as float32 (but save as bool for smaller dsets)
        """
        some_y = tf.cast(some_y, tf.float32)
        return some_X, some_y


    ds = ds.map(convert_y)

    print("---Split---")
    ds = ds.shuffle(ds.cardinality())
    train_samples = int(N_SAMPLES * (1 - VAL_FRAC))
    val_samples = N_SAMPLES - train_samples
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    train_ds = ds.take(train_samples)
    val_ds = ds.skip(train_samples)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    for X, y in train_ds.take(1):
        print("train", X.shape, y.shape)
    for X, y in val_ds.take(1):
        print("val", X.shape, y.shape)
        print(f"val")

    print("---Compile---")
    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.BinaryCrossentropy()
    metrics = [keras.metrics.BinaryAccuracy()]
    model = get_simple_cnn(data_shape, FILTER_SIZES)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )

    # prep output
    model_file = output_dir / "model.weights.h5"
    csv_file = output_dir / "model.csv"
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_folder = tb_dir / "naive_segment" / date_str
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, save_best_only=True),
        keras.callbacks.CSVLogger(csv_file),
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
        keras.callbacks.EarlyStopping(patience=10),
    ]

    print("---Fit---")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_ds,
    )


if __name__ == "__main__":
    main()