"""
Visually inspect the 2D dataset
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cancer_ml.plotting import set_seaborn

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_val15_test15_256-256")

# go!
i_sample = np.random.choice(100)

train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
X, y = next(iter(train_ds.skip(i_sample).take(1)))
print(X.shape)
print(y.shape)

X = np.squeeze(X.numpy())
y = np.squeeze(y.numpy())
y = y.astype(np.float32)
y[y == 0] = np.nan


set_seaborn()
fig, ax = plt.subplots()
ax.imshow(X, vmin=0, vmax=1, cmap="gray")
ax.imshow(y, vmin=0, vmax=1, cmap="Reds", alpha=0.66)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig(DSET_FOLDER / "example.png", bbox_inches="tight")
plt.close()
