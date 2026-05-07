"""
Custom keras callbacks.
"""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb


class PredictionPlotCallback(keras.callbacks.Callback):
    """
    Plot model predictions every n epochs.
    """
    def __init__(
            self,
            model: keras.Model,
            output_folder: Path,
            use_wandb: bool,
            dset: tf.data.Dataset,
            every_n_epoch: int = 10,
            n_images_to_plot: int = 3,
    ) -> None:
        super().__init__()

        # get a batch from which to sample images
        X, y = next(iter(dset.take(1)))
        X = X.numpy()
        y = y.numpy()

        self.pred_model = model
        self.images = X
        self.labels = y
        self.every_n_epoch = every_n_epoch
        self.output_folder = output_folder
        self.n_images_to_plot = n_images_to_plot
        self.use_wandb = use_wandb
        if self.use_wandb:
            print("Using WandB to store plots - output folder will be ignored.")

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Plot predictions if epoch index matches target freq."""
        if epoch % self.every_n_epoch == 0:
            self.plot(epoch)

    def plot(self, epoch: int) -> None:
        """Plot prediction images."""
        n_images = self.images.shape[0]
        i_random = np.random.choice(n_images, size=self.n_images_to_plot)
        data_img = self.images[i_random]
        label_img = self.labels[i_random]
        pred_img = self.pred_model.predict(data_img)
        fig, axes = plt.subplots(
            nrows=self.n_images_to_plot,
            ncols=3,
            layout="constrained",
            figsize=(9, 3 * self.n_images_to_plot),
        )
        fig.suptitle(f"Epoch {epoch}")
        for i_plot in range(self.n_images_to_plot):
            ax = axes[i_plot, 0]
            ax.set_title(f"Input data: {i_random[i_plot]}")
            ax.imshow(np.squeeze(data_img[i_plot]), cmap="Greys")

            ax = axes[i_plot, 1]
            ax.set_title("Ground truth")
            ax.imshow(np.squeeze(label_img[i_plot]), vmin=0, vmax=1, cmap="Reds")

            ax = axes[i_plot, 2]
            ax.set_title("Prediction")
            ax.imshow(np.squeeze(pred_img[i_plot]), vmin=0, vmax=1, cmap="Reds")


        if self.use_wandb:
            wandb.log({"pred_img": wandb.Image(fig)}, step=epoch, commit=False)
        else:
            save_file = self.output_folder / f"epoch{epoch}.png"
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            print(f"{save_file} saved.")
        plt.close()


