from pathlib import Path

import keras

from cancer_ml.models.params import write_hparams


def fit_and_evaluate(
        model: keras.Model,
        dsets: dict,
        model_dir: Path,
        hparams: dict,
        callbacks: list | None = None,
        verbose: int = 1,
) -> None:
    """
    Fit and evaluate a model.
    """
    # set paths
    best_weights_file = model_dir / "best.weights.h5"
    final_weights_file = model_dir / "final.weights.h5"
    csv_file = model_dir / "log.csv"

    # save hparams before
    assert "n_epochs" in hparams.keys()
    write_hparams(hparams, model_dir / "hparams.json")

    # prepare some standard callbacks
    if callbacks is None:
        callbacks = []
    extra_callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_weights_file,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_dice"),
        keras.callbacks.CSVLogger(csv_file),
    ]
    callbacks.extend(extra_callbacks)

    print("---Fit---")
    model.fit(
        dsets["train"],
        validation_data=dsets["val"],
        epochs=hparams["n_epochs"],
        callbacks=callbacks,
        verbose=verbose,
    )
    model.save_weights(final_weights_file)

    print("---Evaluate---")
    model.load_weights(best_weights_file)
    test_loss, test_metrics = model.evaluate(
        dsets["test"],
        verbose=verbose,
    )
    hparams["test_loss"] = test_loss
    hparams["test_dice"] = test_metrics
    write_hparams(hparams, model_dir / "hparams_finished.json")