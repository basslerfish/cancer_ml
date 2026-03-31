"""
Predict cancer mask on 2D image.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cancer_ml.models.two_dims.custom import get_simple_cnn

# params
MODEL_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/2d/flattened")
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/flattened")
FS = [16, 32]
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
model = get_simple_cnn(
    filter_sizes=FS,
    input_shape=input_shape,
)
weights_file = MODEL_DIR / "cnn.weights.h5"
model.load_weights(weights_file)

# predict mask
X = tf.expand_dims(X, axis=0)
y_pred = model.predict(X)
y_pred = np.squeeze(y_pred)


# some formatting
X = np.squeeze(X.numpy())
print(np.min(y_true), np.max(y_true))
y_true[y_true == 0] = np.nan


# plot
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