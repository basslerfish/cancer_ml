from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cancer_ml.models.three_dims import get_simple_cnn


# go!
MODEL_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/3d/32-64_64-128-128-1")
DSET_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/samples500_val15_test15_128-128-64")
I_SAMPLE = 5

# get parms
fs, shape = MODEL_DIR.name.split("_")
fs = fs.split("-")
fs = [int(x) for x in fs]
print(fs)

shape = shape.split("-")
shape = [int(x) for x in shape]
print(shape)

# load dset
train_ds = tf.data.Dataset.load(str(DSET_DIR / "train"))
X, y_true = next(iter(train_ds.skip(I_SAMPLE).take(1)))
print(X.shape)
print(y_true.shape)


# load model
model = get_simple_cnn(
    input_shape=shape,
    filter_sizes=fs,
)
weights_file = MODEL_DIR / "cnn.weights.h5"
model.load_weights(weights_file)

X = tf.expand_dims(X, axis=0)
y_pred = model.predict(X)

# process a little
y_true = y_true.numpy()
y_true = y_true.astype(np.float32)
y_true = np.squeeze(y_true)
y_pred = np.squeeze(y_pred)
y_true[y_true == 0] = np.nan

print(np.min(y_pred), np.max(y_pred))
y_pred = y_pred / np.max(y_pred)

# go
n_seg = np.nansum(y_true, axis=(1, 2))
print(n_seg)
i_section = np.argmax(n_seg)

X = np.squeeze(X.numpy())

fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.imshow(X[i_section, :, :], vmin=-1, vmax=5, cmap="gray")
ax.imshow(y_true[i_section, :, :], vmin=0, vmax=1, cmap="Reds", alpha=0.5)

ax = axes[1]
ax.imshow(X[i_section, :, :], vmin=-1, vmax=5, cmap="gray")
ax.imshow(y_pred[i_section, :, :], vmin=0, vmax=1, cmap="Reds", alpha=0.5)
plt.savefig(MODEL_DIR / "example.png")
plt.close()