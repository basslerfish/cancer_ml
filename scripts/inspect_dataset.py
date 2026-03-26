"""
Let's see how the data has changed after preprocessing.
Would be stupid to make a mistake here already
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from src.load import find_and_load

# paths
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/train_100")
DSET_CSV = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/train_100.csv")
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results")

# load data and sample info
print("---Load---")
ds = tf.data.Dataset.load(str(DSET_FOLDER))
df = pd.read_csv(DSET_CSV)

# load first sample postprocessing
for X, y in ds:
    break

X = X.numpy()
y = y.numpy()

# load first sample preprocessing
org_name = df["sample"].values[0]
print(f"Name of first sample: {org_name}")
org_folder = SOURCE / org_name
org_t1, org_gtv = find_and_load(org_folder)

print("---Plot---")
vmin, vmax = np.percentile(org_t1, [0.5, 99.5])
org_gtv[org_gtv == 0] = np.nan
y[y == 0] = np.nan
fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ax = axes[0]
ax.set_title("Pre")
org_t1_im = ax.imshow(org_t1[:, :, 0],  cmap="gray", vmin=vmin, vmax=vmax)
org_gtv_im = ax.imshow(org_gtv[:, :, 0], vmin=0, vmax=1, cmap="Reds", alpha=0.66)

ax = axes[1]
ax.set_title("Post")
post_t1_im = ax.imshow(X[:, :, 0], vmin=-2, vmax=2, cmap="gray")
post_gtv_im = ax.imshow(y[:, :, 0], vmin=0, vmax=1, cmap="Reds", alpha=0.66)

z_org = org_t1.shape[2]
z_new = X.shape[2]
t = np.linspace(0, z_org - 1, z_new)
t = t.astype(int)


def update(i_frame: int) -> None:
    """Update figure"""
    i_org = t[i_frame]
    org_t1_im.set_array(org_t1[:, :, i_org])
    org_gtv_im.set_array(org_gtv[:, :, i_org])
    axes[0].set_title(f"Pre: {i_org}/{org_t1.shape[2]}")

    post_t1_im.set_array(X[:, :, i_frame])
    post_gtv_im.set_array(y[:, :, i_frame])
    axes[1].set_title(f"Post: {i_frame}/{X.shape[2]}")

ani = FuncAnimation(fig, update, frames=X.shape[-1], interval=50)
ani.save(OUTPUT / f"{org_name}_pre_post.gif")
plt.close()