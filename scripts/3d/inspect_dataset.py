"""
Verify preprocessing visually.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from cancer_ml.plotting import set_seaborn
from cancer_ml.load import find_and_load_sample

# paths
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/3d/samples500_val15_test15_128-128-64")
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")

train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
df = pd.read_csv(DSET_FOLDER / "sample_ids.csv")

# load original
is_train = df["split"] == "train"
i_sample = np.random.choice(df.loc[is_train, "i_sample_in_ds"].values)
is_sample = df["i_sample_in_ds"] == i_sample
org_name = df.loc[is_train & is_sample, "sample"].values[0]
print(f"{org_name=}")

# load original
org_folder = SOURCE / org_name
assert org_folder.is_dir()
t1_imgs, gtv_imgs = find_and_load_sample(org_folder)
print(f"Original shape: {t1_imgs.shape}")
z_pre = t1_imgs.shape[2]
n_seg = np.sum(gtv_imgs, axis=(0, 1))
i_section = np.argmax(n_seg)

# get preprocessed
post_t1, post_gtv = next(iter(train_ds.skip(i_sample).take(1)))
print(f"Preprocessed shape: {post_t1.shape}")
post_t1 = np.squeeze(post_t1.numpy())
post_gtv = np.squeeze(post_gtv.numpy())
z_post = post_t1.shape[0]

# align sections
x_align = np.linspace(0, z_post, z_pre)
x_align = x_align.astype(int)
i_post = x_align[i_section]
print(f"Section: {i_section} in pre -> {i_post} in post")

# plot
set_seaborn()
gtv_imgs[gtv_imgs == 0] = np.nan
fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
fig.suptitle(org_name)
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ax = axes[0]
ax.set_title(f"Pre: section {i_section}/{z_pre}")
vmin, vmax = np.percentile(t1_imgs, [0.5, 99.5])
ax.imshow(t1_imgs[:, :, i_section], cmap="gray", vmin=vmin, vmax=vmax)
ax.imshow(gtv_imgs[:, :, i_section], cmap="Reds", vmin=0, vmax=1)

ax = axes[1]
ax.set_title(f"Post: section {i_post}/{z_post}")
post_gtv = post_gtv.astype(np.float32)
post_gtv[post_gtv == 0] = np.nan
ax.imshow(post_t1[i_post, :, :], cmap="gray", vmin=-1, vmax=5)
ax.imshow(post_gtv[i_post, :, :], cmap="Reds", vmin=0, vmax=1)

plt.savefig(DSET_FOLDER / "example.png")