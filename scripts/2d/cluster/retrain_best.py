"""
Retrain best model after grid search.
Unfinished as of now.
"""
from pathlib import Path

import keras_tuner as kt

from cancer_ml.models.two_dims.search import build_model


def main() -> None:
    raise NotImplementedError()
    TUNER_DIR = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/2d/optimize_2d_dropout")


    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective="val_dice",
        directory=TUNER_DIR.parent,
        project_name="optimize_2d_dropout",
        overwrite=False,
    )
    tuner.reload()

    best_trial = tuner.oracle.get_best_trials(1)[0]
    print(f"Best objective: {best_trial.score:.4f}")


    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(best_hps.values)
    model = tuner.get_best_models(1)[0]

    model.fit(

    )

if __name__ == "__main__":
    pass