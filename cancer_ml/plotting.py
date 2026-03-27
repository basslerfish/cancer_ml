import numpy as np

def plot_overlay(t1_img, gtv_img, ax, vmin=None, vmax=None) -> None:
    if vmin is None:
        vmin, vmax = np.percentile(t1_img, [0.5, 99.5])
    ax.imshow(t1_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(gtv_img, cmap="Reds", vmin=0, vmax=1)