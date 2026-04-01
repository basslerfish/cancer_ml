"""
Let's fit a pretrained segmentation model to our data.

We use DeepLabv3+ here.
This uses an encoder to downsample the image (stride 16).
EG ResNet from 512 x 512 x 3 to 32 x 32 x 2,048
We also save an intermediate stage of downsampling with stride 4.
We use an ASPP head with 5 branches to apply depthwise separable convs at different magnifications.
We also reduce the number of channels in each branch to 256 with 1x1 convs.
We concatenate the ASPP branch outputs, then 1x1 conv to 256 channels

We upsample the ASPP branch output with stride 4.
We 1x1 conv the stride 4 decoder output to reduce channel count.
We concatenate. We apply convolution a few times.
We upsample with stride 4 again. We use a single 1x1 conv layer to get to num classes.
"""
import os
import datetime
from pathlib import Path

import keras
import tensorflow as tf

from cancer_ml.models.params import get_data_params
from cancer_ml.models.two_dims.pretrained import get_pretrained_deeplab
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.base import fit_and_evaluate

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_uint8_val15_test15_128-128")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/")
TB_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/tb_runs/")
N_EPOCHS = 1
BATCH_SIZE = 4

# load data
print("---Load data---")
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


dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    ds = ds.map(preprocess).batch(BATCH_SIZE)
    dsets[name] = ds


# get basic info
X, y = next(iter(dsets["train"].take(1)))
hparams = get_data_params(X)
input_shape = X.shape
print(f"{X.shape=}")
print(f"{y.shape=}")
assert X.shape[-1] == 3  # should be 3 channels for pretrained

# load model
print("---Load model---")
model = get_pretrained_deeplab()
model.preprocessor.image_converter.image_size = (X.shape[1], X.shape[2])

optimizer = keras.optimizers.Adam()
loss_fn = DiceBCELoss()
metrics = [keras.losses.Dice()]
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics,
)

# prep output
hparams["model_type"] = "pretrained_deeplab"
hparams["n_epochs"] = N_EPOCHS
hparams["batch_size"] = BATCH_SIZE

date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = OUTPUT / "2d" / date_str
os.makedirs(model_dir, exist_ok=True)
tb_folder = TB_FOLDER / "2d_pretrained" / date_str
callbacks = [
    keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    keras.callbacks.EarlyStopping(patience=10),
]
fit_and_evaluate(
    model=model,
    dsets=dsets,
    hparams=hparams,
    model_dir=model_dir,
    callbacks=callbacks,
)
