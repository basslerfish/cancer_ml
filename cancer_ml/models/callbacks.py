"""
Custom keras callbacks.
"""
import keras


class UnfreezeCallBack(keras.callbacks.Callback):
    """
    Unfreeze full model at some epoch.
    Update: turns out it's not allowed to recompile during .fit()
    """
    def __init__(
            self,
            model: keras.Model,
            epoch_to_unfreeze: int = 20,
    ) -> None:
        super().__init__()
        self.epoch_to_unfreeze = epoch_to_unfreeze
        self.fit_elements = {
            "model": model,
        }

    def on_epoch_begin(self, epoch, logs=None) -> None:
        if epoch  == self.epoch_to_unfreeze:
            print("Unfreezing all model weights!")
            model = self.fit_elements["model"]
            model.trainable = True
            model.compile(
                optimizer=model.optimizer,
                metrics=model.metrics,
                loss=model.loss,
            )
