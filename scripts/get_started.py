"""
Let's load some data.
There's T1-weighted images with a flexible range.
Then there's

Images: x, y, z

"""
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.load import find_and_load

# paths
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2/")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results")

# get random folder
print("---Choosing folder---")
folders = list(SOURCE.iterdir())
folders = list(filter(lambda x: x.is_dir(), folders))
random_folder = np.random.choice(folders)
print(f"Random folder: {random_folder}")

# load
print("---Loading data---")
t1_data, gtv_data = find_and_load(random_folder)

dtypes = {"t1": t1_data, "gtv": gtv_data}
for name, this_data in dtypes.items():
    min_val = np.min(this_data)
    max_val = np.max(this_data)
    if name == "t1":
        vmin = min_val
        vmax = np.quantile(this_data, 0.99)
    print(f"{name} images: shape={this_data.shape}, {min_val=:.2f}, {max_val=:.2f}")


# plot
print("---Plotting---")
gtv_data[gtv_data == 0] = np.nan  # for plotting
output = OUTPUT / "animations"
os.makedirs(output, exist_ok=True)
fig, ax = plt.subplots()

# initial
t1_im = ax.imshow(t1_data[:, :, 0], animated=True, cmap="gray", vmin=vmin, vmax=vmax)
gtv_im = ax.imshow(gtv_data[:, :, 0], animated=True, cmap="Reds", vmin=0, vmax=1, alpha=0.66)
n_frames = t1_data.shape[2]
fig.suptitle(random_folder.name)
ax.set_title(f"Section 1/{n_frames}")
ax.set_xlabel("x")
ax.set_ylabel("y")


def update(i_frame: int) -> None:
    """Update shown image."""
    new_data = t1_data[:, :, i_frame]
    ax.set_title(f"Section {i_frame + 1}/{n_frames}")
    t1_im.set_array(new_data)
    gtv_im.set_array(gtv_data[:, :, i_frame])

# save
ani = FuncAnimation(fig, update, frames=t1_data.shape[2], interval=50)
save_path = output / f"{random_folder.name}.gif"
ani.save(save_path, writer="pillow")
plt.show()