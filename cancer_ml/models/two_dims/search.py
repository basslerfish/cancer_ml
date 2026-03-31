import keras
import keras_tuner as kt

from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.two_dims.custom import get_advanced_cnn




def build_model(hp: kt.HyperParameters) -> keras.Model:
    filter_sizes = {
        "16-32-64": [16, 32, 64],
        "32-64-128": [32, 64, 128],
        "32-64-128-256": [32, 64, 128, 256],
    }
    input_shape=(128, 128, 1)

    fs_name = hp.Choice("filter_sizes", list(filter_sizes.keys()))
    filter_sizes = filter_sizes[fs_name]
    model = get_advanced_cnn(
        input_shape=input_shape,
        filter_sizes=filter_sizes,
        add_skips=hp.Choice("add_skips", [False, True]),
        dropout_rate=hp.Choice("dropout_rate", values=[0.1, 0.3, 0.5])
    )
    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice(), keras.losses.BinaryCrossentropy()]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    return model