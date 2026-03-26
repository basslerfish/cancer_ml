"""
Let's see how the data has changed after preprocessing.
Would be stupid to make a mistake here already.

The new data shape is (n_img, x, y, n_channels) -> (128, 256, 256, 1)
"""
import os
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
for X, y in ds.take(1):
    break

X = X.numpy()
y = y.numpy()
print(f"T1 data in dataset: {X.shape}")
print(f"Segmentation data in dataset: {y.shape}")
X = np.squeeze(X)  # remove empty channel dim
y = np.squeeze(y)
print(f"Datatype of y: {y.dtype}")
y = y.astype(np.float32)

n_labeled = np.sum(y)
frac_labeled = n_labeled / y.size
print(f"Frac labeled in y: {frac_labeled:.1%}")

# load first sample preprocessing
org_name = df["sample"].values[0]
print(f"Name of first sample: {org_name}")
org_folder = SOURCE / org_name
org_t1, org_gtv = find_and_load(org_folder)

# figure out timeline
z_org = org_t1.shape[2]
z_new = X.shape[0]
t = np.linspace(0, z_org - 1, z_new)
t = t.astype(int)

print("---Plot---")
I_STATIC = 64
output = OUTPUT / "examine"
os.makedirs(output, exist_ok=True)
i_static_org = t[I_STATIC]

vmin, vmax = np.percentile(org_t1, [0.5, 99.5])
org_gtv[org_gtv == 0] = np.nan
y[y == 0] = np.nan
fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ax = axes[0]
ax.set_title(f"Pre: {i_static_org}/{org_t1.shape[2]}")
org_t1_im = ax.imshow(org_t1[:, :, i_static_org],  cmap="gray", vmin=vmin, vmax=vmax)
org_gtv_im = ax.imshow(org_gtv[:, :, i_static_org], vmin=0, vmax=1, cmap="Reds", alpha=0.66)

ax = axes[1]
ax.set_title(f"Post: {I_STATIC}/{X.shape[0]}")
post_t1_im = ax.imshow(X[I_STATIC, :, :], vmin=-2, vmax=2, cmap="gray")
post_gtv_im = ax.imshow(y[I_STATIC, :, :], vmin=0, vmax=1, cmap="Reds", alpha=0.66)

plt.savefig(output / f"{org_name}_pre_post.png")

print("---Animate---")

def update(i_frame: int) -> None:
    """Update figure"""
    i_org = t[i_frame]
    org_t1_im.set_array(org_t1[:, :, i_org])
    org_gtv_im.set_array(org_gtv[:, :, i_org])
    axes[0].set_title(f"Pre: {i_org}/{org_t1.shape[2]}")

    post_t1_im.set_array(X[i_frame, :, :])
    post_gtv_im.set_array(y[i_frame, :, :])
    axes[1].set_title(f"Post: {i_frame}/{X.shape[2]}")

ani = FuncAnimation(fig, update, frames=X.shape[0], interval=50)
ani.save(output / f"{org_name}_pre_post.gif")
plt.close()