"""
Retrain best model after grid search.
"""
import datetime
import os

import keras
import keras_tuner as kt
import tensorflow as tf

from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.loss import DiceBCELoss
from cancer_ml.models.params import get_data_params
from cancer_ml.models.two_dims.custom import get_advanced_cnn
from cancer_ml.models.two_dims.search import build_model
from cancer_ml.utils import get_args_dirs

# params
BATCH_SIZE = 128
N_EPOCHS = 100


def main() -> None:
    # paths
    data_dir, output_dir, tb_dir = get_args_dirs()

    # load hparams search
    print("---Load search result---")
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective="val_dice",
        directory=output_dir,
        project_name="optimize_advanced",
        overwrite=False,
    )
    tuner.reload()
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_score = best_trial.score
    print(f"Best objective: {best_score:.4f}")
    best_hps = tuner.get_best_hyperparameters(1)[0].values
    print("Best parameters:")
    for k, v in best_hps.items():
        print(f"\t {k} -> {v}")
    best_hps["filter_sizes"] = [int(x) for x in best_hps["filter_sizes"].split("-")]

    print("---Load data---")
    def change_dtype(t1_imgs, gtv_imgs) -> tuple:
        """We want X as float16 (to save memory) and y as float32 (to have good loss calculations)"""
        t1_imgs = tf.cast(t1_imgs, tf.float16)
        gtv_imgs = tf.cast(gtv_imgs, tf.float32)
        return t1_imgs, gtv_imgs

    dsets = {}
    for name in ["train", "val", "test"]:
        ds = tf.data.Dataset.load(str(data_dir / name))
        ds = ds.map(change_dtype).batch(BATCH_SIZE)
        dsets[name] = ds
    X, y = next(iter(dsets["train"].take(1)))
    hparams = get_data_params(X)

    # set up model
    model = get_advanced_cnn(
        input_shape=X.shape[1:],
        filter_sizes=best_hps["filter_sizes"],
        dropout_rate=best_hps["dropout_rate"],
        add_skips=best_hps["add_skips"],
    )
    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )

    # prep output
    print("---Prepping output---")
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / date_str
    os.makedirs(model_dir, exist_ok=True)

    # set up callbacks
    tb_folder = tb_dir / "2d" / date_str
    callbacks = [
        keras.callbacks.TensorBoard(tb_folder, update_freq="epoch"),
    ]

    # save hyperparameters
    hparams["model_type"] = "advanced"
    hparams["comment"] = "post_search"
    hparams["search_score"] = best_score
    hparams["filter_sizes"] = best_hps["filter_sizes"]
    hparams["dropout_rate"] = best_hps["dropout_rate"]
    hparams["n_epochs"] = N_EPOCHS
    hparams["batch_size"] = BATCH_SIZE
    hparams["add_skips"] = best_hps["add_skips"]


    # fit
    fit_and_evaluate(
        model=model,
        hparams=hparams,
        dsets=dsets,
        model_dir=model_dir,
        callbacks=callbacks,
        verbose=2,
    )

if __name__ == "__main__":
    main()