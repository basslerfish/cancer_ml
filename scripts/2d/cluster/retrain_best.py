"""
Retrain best model after grid search.
Unfinished as of now.
"""
import keras
import keras_tuner as kt
import tensorflow as tf

from cancer_ml.models.two_dims.search import build_model
from cancer_ml.utils import get_args_dirs
from cancer_ml.models.base import fit_and_evaluate
from cancer_ml.models.loss import DiceBCELoss


BATCH_SIZE = 128


def main() -> None:
    # paths
    data_dir, output_dir, tb_dir = get_args_dirs()

    # load hparams search
    print("---Load search result---")
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective="val_dice",
        directory=output_dir,
        project_name="optimize_2d_dropout",
        overwrite=False,
    )
    tuner.reload()
    best_trial = tuner.oracle.get_best_trials(1)[0]
    print(f"Best objective: {best_trial.score:.4f}")
    best_hps = tuner.get_best_hyperparameters(1)[0]

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

    # set up model
    model = tuner.get_best_models(1)[0]
    optimizer = keras.optimizers.Adam()
    loss_fn = DiceBCELoss()
    metrics = [keras.losses.Dice]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )

    # fit
    fit_and_evaluate(
        model=model,
        hparams={},
        dsets={},
        model_dir=None,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    pass