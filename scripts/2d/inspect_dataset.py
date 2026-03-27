"""
Visually inspect the 2D dataset
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/flattened")

# go!
train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
X, y = next(iter(train_ds.take(1)))
print(X.shape)
print(y.shape)

X = np.squeeze(X.numpy())
y = np.squeeze(y.numpy())
y = y.astype(np.float32)
y[y == 0] = np.nan

fig, ax = plt.subplots()
ax.imshow(X, vmin=-1, vmax=5, cmap="gray")
ax.imshow(y, vmin=0, vmax=1, cmap="Reds", alpha=0.66)
plt.savefig(DSET_FOLDER / "example.png")
plt.close()
