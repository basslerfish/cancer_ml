from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.animation import FuncAnimation


PATH = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/train_100")

ds = tf.data.Dataset.load(str(PATH))

for X, y in ds:
    break




fig, ax = plt.subplots()
im = ax.imshow(X[:, :, 0], vmin=-3, vmax=3, cmap="gray")


def update(i_frame):
    im.set_array(X[:, :, i_frame])

ani = FuncAnimation(fig, update, frames=X.shape[-1], interval=50)
plt.show()