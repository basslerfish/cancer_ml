import keras

class TverskyBCELoss(keras.losses.Loss):
    """
    Tversyk loss is similar to Dice but allows weighing of classes.
    """
    def __init__(
            self,
            tversky_weight: float = 0.8,
            beta: float = 0.8,
            **kwargs,
    ) -> None:
        super().__init__(name="TverskyBCELoss", **kwargs)
        self.tversky_weight = tversky_weight
        self.bce_weight = 1 - tversky_weight
        alpha = 1 - beta
        self.tversyk = keras.losses.Tversky(alpha=alpha, beta=beta)

    def call(self, y_true, y_pred):
        tversky_loss = self.tversyk(y_true, y_pred)
        bce_loss = keras.losses.BinaryCrossentropy()(y_true, y_pred)
        combined_loss = self.dice_weight * tversky_loss + self.bce_weight * bce_loss
        return combined_loss


class DiceBCELoss(keras.losses.Loss):
    """
    Combined Dice and BCE loss.
    BCE loss on segmentation is not very good predictor of performance when class imbalance is strong.
    Dice is a better fit, but may be so bad at beginning that we supplement with BCE.
    """
    def __init__(self, dice_weight: float = 0.5, **kwargs) -> None:
        super().__init__(name="DiceBCELoss", **kwargs)
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight

    def call(self, y_true, y_pred):
        dice_loss = keras.losses.Dice()(y_true, y_pred)
        bce_loss = keras.losses.BinaryCrossentropy()(y_true, y_pred)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return combined_loss
