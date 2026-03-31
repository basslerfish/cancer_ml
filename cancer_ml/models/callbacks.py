"""
Custom keras callbacks.
"""
import keras

class UnfreezeCallBack(keras.callbacks.Callback):
    """
    Unfreeze full model at some epoch.

    """
    def __init__(
            self,
            model: keras.Model,
            loss_fn: keras.losses.Loss,
            optimizer: keras.optimizers.Optimizer,
            metrics: list,
            epoch_to_unfreeze: int = 20,
    ) -> None:
        super().__init__()
        self.epoch_to_unfreeze = epoch_to_unfreeze
        self.fit_elements = {
            "model": model,
            "loss": loss_fn,
            "optimizer": optimizer,
            "metrics": metrics,
        }

    def on_epoch_begin(self, epoch, logs=None) -> None:
        if epoch  == self.epoch_to_unfreeze:
            print("Unfreezing all model weights!")
            model = self.fit_elements["model"]
            model.trainable = True
            model.compile(
                optimizer=self.fit_elements["optimizer"],
                metrics=self.fit_elements["metrics"],
                loss=self.fit_elements["loss"]
            )
