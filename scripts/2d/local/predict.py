"""
Predict cancer mask on 2D image.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cancer_ml.models.params import read_hparams
from cancer_ml.models.two_dims.custom import get_advanced_cnn, get_simple_cnn
from cancer_ml.plotting import set_seaborn

# params
MODEL_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/2d/20260401_160905")
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_zscore_val15_test15_128-128")
I_SAMPLE = 100

# load data
dsets = {}
for name in ["train", "val", "test"]:
    ds = tf.data.Dataset.load(str(DSET_FOLDER / name))
    dsets[name] = ds

X, y_true = next(iter(dsets["train"].skip(I_SAMPLE).take(1)))
input_shape = X.shape
y_true = np.squeeze(y_true.numpy()).astype(np.float32)

# load model
hparams = read_hparams(MODEL_DIR / "hparams.json")
model_type = hparams["model_type"]
if model_type == "simple":
    model = get_simple_cnn(
        filter_sizes=hparams["filter_sizes"],
        input_shape=input_shape,
    )
elif model_type == "advanced":
    raise NotImplementedError(f"{model_type=}")
else:
    raise ValueError(f"{model_type=}")
weights_file = MODEL_DIR / "best.weights.h5"
model.load_weights(weights_file)

# predict mask
X = tf.expand_dims(X, axis=0)
y_pred = model.predict(X)
y_pred = np.squeeze(y_pred)

# some formatting
X = np.squeeze(X.numpy())
y_true[y_true == 0] = np.nan

# plot
set_seaborn()
fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
ax = axes[0]
ax.set_title("True mask")
ax.imshow(X, vmin=-1, vmax=5, cmap="gray")
ax.imshow(y_true, vmin=0, vmax=1, cmap="Reds", alpha=0.5)
ax = axes[1]
ax.set_title("Predicted mask")
ax.imshow(X, vmin=-1, vmax=5, cmap="gray")
ax.imshow(y_pred, vmin=0, vmax=1, cmap="Reds", alpha=0.5)
plt.savefig(MODEL_DIR / "example_predict.png", dpi=300, bbox_inches="tight")
plt.close(fig)